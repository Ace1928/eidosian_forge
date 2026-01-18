import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
class _SplitterBase:
    """
    Splits a GraphModule into sub-GraphModules for execution on CPU or the accelerator.
    Output is a GraphModule with supported and unsupported operators grouped into as few sub-GraphModules as possible.
    Assumes that only "call_module", "call_function" and "call_method" from FX IR can potentially be executed on the accelerator.

    Given the following graph:
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    and providing "operator_support" that indicates that 'b' and 'c' can be executed on the accelerator,
    we will get the following split result:

    main:
    def forward(self, a):
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        getitem = run_on_acc_0_0[0]
        getitem_1 = run_on_acc_0_0[1]
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        sin_1 = torch.sin(a)
        cos_1 = torch.cos(a)
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        add_1 = sin_1 + cos_1
        return add_1
    """
    PCIe_BW = 100 * 2 ** 30

    def __init__(self, module: torch.fx.GraphModule, sample_input: Sequence[Any], operator_support: OperatorSupportBase, settings: _SplitterSettingBase, non_acc_submodule_name: str='_run_on_cpu_'):
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        assert isinstance(module, torch.fx.GraphModule)
        self.module = module
        ShapeProp(self.module).propagate(*sample_input)
        self.settings = settings
        self.operator_support = operator_support
        self.sample_input = sample_input
        self.acc_nodes = FxNetAccNodesFinder(self.module, self.operator_support, self.settings.allow_non_tensor)()
        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, self.acc_nodes)()
        self.deps = self.find_deps()
        self.update_deps_for_fusions()
        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: Dict[str, str] = {}

    def get_node_submodule_map(self) -> Dict[str, str]:
        """ Returns a map from node name to submodule name, e.g.
            node: main_module_impl_impl_over_arch_unary_multiple_embedding
              _pooling_embedding_pooling_sparse_entity_equivalence_key
              _proxy_embedding_bag
            maps to submodule name of: _run_on_acc_1
        """
        return self._node_submodule_map

    def find_deps(self) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
        deps: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            for user in node.users:
                if user.op != 'output':
                    deps[user].add(node)
        return deps

    def update_deps_for_fusions(self):
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
        for node in self.fusions:
            fusion = self.fusions[node]
            for fused_neighbor in fusion:
                self.deps[node].update(self.deps[fused_neighbor] - fusion)
                for user in fused_neighbor.users:
                    if user not in fusion:
                        self.deps[user].add(node)

    def _lower_model_to_backend(self, mod: torch.fx.GraphModule, inputs: Tensors) -> torch.nn.Module:
        """
        Lower the model to a backend.
        """
        return mod

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors) -> str:
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """
        return 'Unable to find a culprit because _find_culprit() function is not implemented.'

    def _draw_graph_based_on_node_support(self, mod: torch.fx.GraphModule, supported_nodes: NodeList):
        color_map = {'default': 'AliceBlue', 'supported': 'chartreuse1', 'unsupported': 'crimson'}

        class CustomDrawer(FxGraphDrawer):

            def _get_node_style(self, node):
                template = super()._get_node_style(node)
                if node in supported_nodes:
                    template['fillcolor'] = color_map['supported']
                elif node.op in CALLABLE_NODE_OPS:
                    template['fillcolor'] = color_map['unsupported']
                else:
                    template['fillcolor'] = color_map['default']
                return template
        drawer = CustomDrawer(mod, 'node_support', ignore_getattr=True)
        dot_graph = drawer.get_main_dot_graph()
        dot_graph.write_raw('node_support.dot')

    def node_support_preview(self, dump_graph: bool=False):
        submodules = dict(self.module.named_modules())
        supported_nodes: NodeList = []
        supported_node_types = defaultdict(set)
        unsupported_node_types = defaultdict(set)

        def get_dtype(arg):
            tensor_meta = arg.meta.get('tensor_meta')
            return getattr(tensor_meta, 'dtype', None)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            target = get_node_target(submodules, node)
            arg_dtypes = [get_dtype(arg) if isinstance(arg, torch.fx.Node) else None for arg in node.args]
            last_index = len(arg_dtypes) - next((i for i, dtype in enumerate(reversed(arg_dtypes)) if dtype is not None), len(arg_dtypes))
            arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
            kwarg_dtypes_tuple = tuple(((k, get_dtype(arg)) for k, arg in node.kwargs.items() if isinstance(arg, torch.fx.Node)))
            if self.operator_support.is_node_supported(submodules, node):
                supported_nodes.append(node)
                supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
            else:
                unsupported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
        if dump_graph:
            self._draw_graph_based_on_node_support(self.module, supported_nodes)
        reports = '\nSupported node types in the model:\n'
        for t, dtypes in supported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f'{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n'
        reports += '\nUnsupported node types in the model:\n'
        for t, dtypes in unsupported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f'{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n'
        print(reports)
        return reports

    def split_preview(self, dump_graph: bool=False):
        reports = ''
        subgraphs = self.put_nodes_into_subgraphs()
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f'Before removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:'
        reports += f' {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n'
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f'After removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:'
        reports += f' {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n'
        for i, subgraph in enumerate(subgraphs):
            reports += f'_run_on_acc_{i}: ' if subgraph.is_acc else f'{self.non_acc_submodule_name}{i}: '
            reports += f'{len(subgraph.nodes)} node(s)\n'
        self.tag(subgraphs)
        split_mod = self.split(remove_tag=True)
        split_mod.eval()
        if dump_graph:
            drawer = FxGraphDrawer(split_mod, 'preview', ignore_getattr=True)
            dot_graphs = drawer.get_all_dot_graphs()
            for name, dot_graph in dot_graphs.items():
                dot_graph.write_raw(f'{name}.dot')
        max_qps: float = self.PCIe_BW
        bottleneck_module = ''
        for node in split_mod.graph.nodes:
            if node.op == 'call_module' and 'acc' in node.target:
                reports += f'\nProcessing acc submodule {node.target}\n'
                submod = getattr(split_mod, node.target)

                def get_submod_inputs(main_mod, submod, example_inputs):
                    sub_inputs = None

                    def get_inputs(self, inputs):
                        nonlocal sub_inputs
                        sub_inputs = inputs
                    handle = submod.register_forward_pre_hook(get_inputs)
                    main_mod(*example_inputs)
                    handle.remove()
                    return sub_inputs
                submod_inputs = get_submod_inputs(split_mod, submod, self.sample_input)
                ShapeProp(submod).propagate(*submod_inputs)
                total_input_bytes = 0
                total_output_bytes = 0
                reports += 'Checking inputs...\n'
                for n in submod.graph.nodes:
                    if n.op == 'placeholder':
                        if not is_node_output_tensor(n):
                            reports += f'Input {n.name} is not a tensor, this might cause problems during lowering!\n'
                        else:
                            total_input_bytes += get_size_of_node(submod, n)[0]
                    if n.op == 'output':
                        output_node = n
                reports += 'Checking outputs...\n'

                def get_bytes(node: torch.fx.Node):
                    nonlocal total_output_bytes
                    nonlocal reports
                    if not is_node_output_tensor(node):
                        reports += f'Output {node.name} is not a tensor, this might cause problems during lowering!\n'
                    else:
                        total_output_bytes += get_size_of_node(submod, node)[0]
                map_arg(output_node.args, get_bytes)
                qps = self.PCIe_BW / max(total_input_bytes, total_output_bytes)
                reports += f'Total input size in bytes is {total_input_bytes}, total output size in bytes is {total_output_bytes},'
                reports += f' theoretical max qps (bounds by PCIe bandwidth) for this submodule is {qps}.\n'
                if qps < max_qps:
                    max_qps = qps
                    bottleneck_module = node.target
                try:
                    lowered_submod = self._lower_model_to_backend(submod, submod_inputs)
                except RuntimeError:
                    reports += 'Run into an error during lowering!\n'
                    reports += self._find_culprit(submod, submod_inputs)
                    continue
                try:
                    lowered_submod(*submod_inputs)
                except RuntimeError:
                    reports += 'Run into an error during inference!\n'
                    reports += self._find_culprit(submod, submod_inputs)
                else:
                    reports += 'Lowering and running succeed!\n'
        reports += f'\nTheoretical max qps (bounds by PCIe bandwidth) for this model is {max_qps},'
        reports += f' bottleneck is submodule {bottleneck_module}.'
        print(reports)
        return reports

    def find_reverse_deps(self, tag_id: Optional[int]=None) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
        result: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue
                if tag_id is None or int(user.tag.split('_')[-1]) < tag_id:
                    result[node].add(user)
        return result

    def update_reverse_deps_for_fusions(self, deps: Dict[torch.fx.Node, NodeSet]):
        processed_node = set()
        for node, fusion in self.fusions.items():
            if node in processed_node:
                continue
            new_dep = set()
            for n in fusion:
                new_dep.update(deps[n])
            new_dep.difference_update(fusion)
            for n in fusion:
                deps[n] = new_dep
                for arg in n.all_input_nodes:
                    if arg not in fusion:
                        deps[arg].update(fusion)
                processed_node.add(n)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
        parent_nodes = set()
        for node in self.module.graph.nodes:
            if node.op in CALLABLE_NODE_OPS and node.tag == tag:
                for arg in node.all_input_nodes:
                    if arg.op in CALLABLE_NODE_OPS and arg.tag != tag:
                        parent_nodes.add(arg)
        return parent_nodes

    def extend_acc_subgraph(self, tag: str):
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
        deps = self.find_reverse_deps(tag_id=int(tag.split('_')[-1]))
        self.update_reverse_deps_for_fusions(deps)
        parent_nodes = self.find_parent_nodes_of_subgraph(tag)
        visited_nodes: NodeSet = set()
        while parent_nodes:
            node = None
            for n in parent_nodes:
                if deps[n] <= visited_nodes and n in self.acc_nodes:
                    node = n
                    break
            if node is None:
                break
            node.tag = tag
            parent_nodes.remove(node)
            visited_nodes.add(node)
            if node in self.fusions:
                for fusion_node in self.fusions[node]:
                    if fusion_node not in visited_nodes:
                        parent_nodes.add(fusion_node)
            for arg in node.all_input_nodes:
                if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                    parent_nodes.add(arg)

    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()
        for node in self.module.graph.nodes:
            if node.op not in {'placeholder', 'get_attr'}:
                continue
            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)
        return (starter_cpu_nodes, starter_acc_nodes)

    def put_nodes_into_subgraphs(self) -> List[Subgraph]:
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()
        acc_subgraph: bool = not any((len(self.deps[n]) == 0 for n in current_cpu_nodes))
        current_subgraph_nodes: NodeList = []
        subgraphs: List[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next((n for n in current_nodes if self.deps[n] <= visited_nodes), None)
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")
                subgraphs.append(Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes))
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue
            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)
        if current_subgraph_nodes:
            subgraphs.append(Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes))
        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")
        return subgraphs

    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
        result: List[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                    result.append(subgraph)
                else:
                    print(f"Eliminating acc subgraph because it's smaller than the threshold: {len(subgraph.nodes)} < {self.settings.min_acc_module_size}")
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        subgraph.is_acc = False
                        result.append(subgraph)
            elif result and (not result[-1].is_acc):
                result[-1].nodes.extend(subgraph.nodes)
            else:
                result.append(subgraph)
        return result

    def tag(self, subgraphs: List[Subgraph]):
        self.tags: List[str] = []
        for subgraph in subgraphs:
            tag = f'_run_on_acc_{len(self.tags)}' if subgraph.is_acc else f'{self.non_acc_submodule_name}{len(self.tags)}'
            self.tags.append(tag)
            for node in subgraph.nodes:
                if hasattr(node, 'tag'):
                    raise FxNetSplitterInternalError(f'Node {node} was already tagged')
                node.tag = tag
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool=False) -> torch.fx.GraphModule:
        split_module = split_by_tags(self.module, self.tags)
        if remove_tag:
            for node in self.module.graph.nodes:
                if hasattr(node, 'tag'):
                    del node.tag
        return split_module

    def __call__(self) -> torch.fx.GraphModule:
        subgraphs = self.put_nodes_into_subgraphs()
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_count = len([s for s in subgraphs if s.is_acc])
        non_acc_subgraphs_count = len(subgraphs) - acc_subgraphs_count
        print(f'Got {acc_subgraphs_count} acc subgraphs and {non_acc_subgraphs_count} non-acc subgraphs')
        self.tag(subgraphs)
        return self.split()

    def generate_split_results(self) -> SplitResult:
        split_module = self()
        submodule_names = []
        for name, mod in split_module.named_children():
            submodule_names.append(name)
        submodule_inputs = generate_inputs_for_submodules(split_module, self.sample_input, submodule_names)
        return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)