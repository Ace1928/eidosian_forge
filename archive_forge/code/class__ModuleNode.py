from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
class _ModuleNode(_IRNode):
    """Representing a sequence of fx.Nodes to be formed into a fx.GraphModule.

    This class encapsulates metadata and provides building block methods to construct this
    layered abstraction from a sequence of flat fx.Nodes.

    Attributes:
    - _stack_meta: Metadata of the module stack.
    - _nodes: List of IR nodes in the module.
    - _reference_root_module: Reference to the root flat fx.GraphModule instance.
    """

    def __init__(self, reference_root_module: torch.fx.GraphModule, stack_meta: _ModuleStackMeta):
        self._stack_meta = stack_meta
        self._nodes: List[_IRNode] = []
        self._reference_module = reference_root_module

    @property
    def stack_meta(self) -> _ModuleStackMeta:
        return self._stack_meta

    @property
    def stack_trace(self) -> Optional[str]:
        assert self._nodes
        return self._nodes[0].stack_trace

    def __str__(self) -> str:
        return f'ModuleNode({self._stack_meta})'

    def is_same_module_as(self, node: _IRNode) -> bool:
        """Determines if the provided node pertains to the same module as this node."""
        return self.stack_meta == node.stack_meta

    def is_parent_module_of(self, node: _IRNode) -> bool:
        """Determines if this node represents a parent module of the provided node."""
        return node.stack_meta.is_superset_of(self.stack_meta)

    def add_leaf_node(self, leaf_node: _LeafNode) -> None:
        """Adds a leaf node to the module.

        The leaf node must belong to the same or a child module. This method will recursively
        construct _ModuleNode instance based on the stack_meta information of the leaf node.
        """
        if self.is_same_module_as(leaf_node) or leaf_node.fx_op == 'call_module':
            self._nodes.append(leaf_node)
        elif self.is_parent_module_of(leaf_node):
            last_node = self._nodes[-1] if self._nodes else None
            if isinstance(last_node, _ModuleNode) and (last_node.is_parent_module_of(leaf_node) or last_node.is_same_module_as(leaf_node)):
                last_node.add_leaf_node(leaf_node)
            else:
                stack_meta = copy.deepcopy(self.stack_meta)
                stack_meta.push(leaf_node.stack_meta[len(self.stack_meta)])
                last_node = _ModuleNode(self._reference_module, stack_meta)
                self._nodes.append(last_node)
                last_node.add_leaf_node(leaf_node)
        else:
            raise AssertionError(f'Node {leaf_node} ({leaf_node.stack_meta}) does not belong to module {self._stack_meta}.')

    def fx_nodes(self) -> Generator[torch.fx.Node, None, None]:
        """Returns an iterator for the sequence of fx nodes this instance holds."""
        for node in self._nodes:
            if isinstance(node, _ModuleNode):
                yield from node.fx_nodes()
            else:
                assert isinstance(node, _LeafNode)
                yield node.fx_node

    def module_inputs(self) -> Sequence[torch.fx.Node]:
        """Extract module inputs from the sequence of fx nodes this instance holds.

        All node args that are produced by nodes outside of the module are considered module
        inputs. The order of returned module inputs is the same as the their use order.

        ### Known limitations

        The original ordering of module inputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module inputs.
        """
        nodes = list(self.fx_nodes())
        assert len(nodes) > 0, 'Cannot extract module inputs from empty nodes.'
        module_inputs: Dict[torch.fx.Node, None] = {}
        node_set: Set[torch.fx.Node] = set(nodes)

        def _extract_arg_if_node_outside_module(arg: Any):
            if isinstance(arg, torch.fx.Node) and arg not in node_set:
                module_inputs[arg] = None
        for node in nodes:
            pytree.tree_map(_extract_arg_if_node_outside_module, node.args)
            pytree.tree_map(_extract_arg_if_node_outside_module, node.kwargs)
        return list(module_inputs.keys())

    def module_outputs(self) -> Sequence[torch.fx.Node]:
        """Extract module outputs from the sequence of fx nodes this instance holds.

        All nodes that are used by nodes outside of the module are considered module
        outputs. The order of returned module outputs is the same as the their creation order.

        ### Known limitations

        The original ordering of module outputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module outputs.
        """
        nodes = list(self.fx_nodes())
        assert len(nodes) > 0, 'Cannot extract module inputs from empty nodes.'
        module_outputs: Dict[torch.fx.Node, None] = {}
        node_set: Set[torch.fx.Node] = set(nodes)
        for node in nodes:
            if any((user not in node_set for user in node.users)):
                module_outputs[node] = None
        return list(module_outputs.keys())

    def build_module(self, module_names: Dict[str, int]) -> torch.fx.GraphModule:
        """
        Constructs the fx.GraphModule for this node, registering submodules as necessary.

        Args:
            module_names: A dictionary of module names and their counts. This is used to
                generate unique module names for submodules. This should be an empty
                dictionary when the method is called on a root module.
        """
        module_class_name = self._stack_meta.qualified_module_class_name
        fx_graph = torch.fx.Graph()
        copy_env: Dict[torch.fx.Node, torch.fx.Node] = {}

        def _arg_transform(node: torch.fx.Node) -> torch.fx.Node:
            return copy_env[node]
        ref_inputs = self.module_inputs()
        for node in ref_inputs:
            copy_env[node] = fx_graph.placeholder(node.name, node.type)
            copy_env[node].meta = copy.copy(node.meta)
        for ir_node in self._nodes:
            if isinstance(ir_node, _LeafNode):
                fx_node = ir_node.fx_node
                copy_env[fx_node] = fx_graph.node_copy(fx_node, arg_transform=_arg_transform)
                continue
            assert isinstance(ir_node, _ModuleNode)
            submodule = ir_node.build_module(module_names)
            ref_submodule_inputs = ir_node.module_inputs()
            ref_submodule_outputs = ir_node.module_outputs()
            unique_submodule_name = _get_unique_module_name(module_names, ir_node.stack_meta.module_display_name)
            self._reference_module.add_submodule(unique_submodule_name, submodule)
            submodule_node = fx_graph.call_module(unique_submodule_name, tuple((_arg_transform(node) for node in ref_submodule_inputs)))
            if len(ref_submodule_outputs) > 1:
                submodule_node.meta['val'] = tuple((ref_output.meta.get('val') for ref_output in ref_submodule_outputs))
                for i, ref_output in enumerate(ref_submodule_outputs):
                    getitem_node = fx_graph.call_function(operator.getitem, args=(submodule_node, i), type_expr=ref_output.type)
                    getitem_node.meta = copy.copy(ref_output.meta)
                    getitem_node.meta['nn_module_stack'] = copy.copy(ref_output.meta['nn_module_stack'])
                    getitem_node.meta['nn_module_stack'].popitem()
                    copy_env[ref_output] = getitem_node
            else:
                copy_env[ref_submodule_outputs[0]] = submodule_node
                submodule_node.meta = copy.copy(ref_submodule_outputs[0].meta)
            if (stack_trace := ir_node.stack_trace) is not None:
                submodule_node.meta['stack_trace'] = stack_trace
            raw_module_stack_meta = ir_node.stack_meta.raw_meta
            assert raw_module_stack_meta is not None
            submodule_node.meta['nn_module_stack'] = copy.copy(raw_module_stack_meta)
            submodule_node.meta['nn_module_stack'].popitem()
        new_nodes = fx_graph.nodes
        if next(iter(reversed(new_nodes))).op != 'output':
            ref_submodule_outputs = self.module_outputs()
            new_outputs = [copy_env[ref_output] for ref_output in self.module_outputs()]
            node = fx_graph.output(new_outputs[0] if len(new_outputs) == 1 else new_outputs)
        graph_module = torch.fx.GraphModule(self._reference_module, fx_graph, module_class_name)
        if (module_class := self._stack_meta.module_class) is not None:
            graph_module.meta['onnx'] = _pass.GraphModuleOnnxMeta(_pass.PackageInfo.from_python_class(module_class))
        return graph_module