from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@dataclasses.dataclass
class GraphInfo:
    """GraphInfo contains validation information of a TorchScript graph and its converted ONNX graph."""
    graph: torch.Graph
    input_args: Tuple[Any, ...]
    params_dict: Dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(default_factory=_experimental.ExportOptions)
    mismatch_error: Optional[AssertionError] = dataclasses.field(default=None, init=False)
    pt_outs: Optional[Sequence[_NumericType]] = dataclasses.field(default=None, init=False)
    upper_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    lower_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    id: str = dataclasses.field(default='')
    _onnx_graph: Optional[torch.Graph] = dataclasses.field(init=False, default=None)
    _EXCLUDED_NODE_KINDS: FrozenSet[str] = frozenset({'prim::Constant', 'prim::ListConstruct', 'aten::ScalarImplicit'})

    def clear(self):
        """Clear states and results of previous verification."""
        self.mismatch_error = None
        self.pt_outs = None
        self._onnx_graph = None
        self.upper_graph_info = None
        self.lower_graph_info = None

    def pretty_print_tree(self):
        """Pretty print `GraphInfo` tree.

        Each node represents a subgraph, showing the number of nodes in the subgraph and
        a check mark if the subgraph has output mismatch between torch and ONNX.

        The id of the subgraph is shown under the node. The `GraphInfo` object for any
        subgraph can be retrieved by calling `graph_info.find_partition(id)`.

        Example::

            ==================================== Tree: =====================================
            5 X   __2 X    __1 ✓
            id:  |  id: 0 |  id: 00
                 |        |
                 |        |__1 X (aten::relu)
                 |           id: 01
                 |
                 |__3 X    __1 ✓
                    id: 1 |  id: 10
                          |
                          |__2 X     __1 X (aten::relu)
                             id: 11 |  id: 110
                                    |
                                    |__1 ✓
                                       id: 111
            =========================== Mismatch leaf subgraphs: ===========================
            ['01', '110']
            ============================= Mismatch node kinds: =============================
            {'aten::relu': 2}

        """
        GraphInfoPrettyPrinter(self).pretty_print()

    def pretty_print_mismatch(self, graph: bool=False):
        """Pretty print details of the mismatch between torch and ONNX.

        Args:
            graph: If True, print the ATen JIT graph and ONNX graph.
        """
        print(f' Mismatch info for graph partition {self.id}: '.center(80, '='))
        if graph:
            print(' ATen JIT graph '.center(80, '='))
            print(self.graph)
            if self._onnx_graph is not None:
                print(' ONNX graph '.center(80, '='))
                print(self._onnx_graph)
        if self.has_mismatch():
            print(' Mismatch error '.center(80, '='))
            print(self.mismatch_error)
        else:
            print(' No mismatch '.center(80, '='))

    @_beartype.beartype
    def has_mismatch(self) -> bool:
        """Return True if the subgraph has output mismatch between torch and ONNX."""
        return self.mismatch_error is not None

    @_beartype.beartype
    def essential_node_count(self) -> int:
        """Return the number of nodes in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        return sum((1 for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS))

    @_beartype.beartype
    def essential_node_kinds(self) -> Set[str]:
        """Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        return {n.kind() for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS}

    @_beartype.beartype
    def all_mismatch_leaf_graph_info(self) -> List['GraphInfo']:
        """Return a list of all leaf `GraphInfo` objects that have mismatch."""
        if not self.has_mismatch():
            return []
        no_mismatch_children = (self.upper_graph_info is None or not self.upper_graph_info.has_mismatch()) and (self.lower_graph_info is None or not self.lower_graph_info.has_mismatch())
        if no_mismatch_children:
            return [self]
        results = []
        if self.upper_graph_info is not None:
            results += self.upper_graph_info.all_mismatch_leaf_graph_info()
        if self.lower_graph_info is not None:
            results += self.lower_graph_info.all_mismatch_leaf_graph_info()
        return results

    @_beartype.beartype
    def find_partition(self, id: str) -> Optional['GraphInfo']:
        """Find the `GraphInfo` object with the given id."""
        if id == self.id:
            return self
        current_length = len(self.id)
        if len(id) > current_length:
            if id[current_length] == '0' and self.upper_graph_info is not None:
                return self.upper_graph_info.find_partition(id)
            elif id[current_length] == '1' and self.lower_graph_info is not None:
                return self.lower_graph_info.find_partition(id)
        return None

    @_beartype.beartype
    def export_repro(self, repro_dir: Optional[str]=None, name: Optional[str]=None) -> str:
        """Export the subgraph to ONNX along with the input/output data for repro.

        The repro directory will contain the following files::

            dir
            ├── test_<name>
            │   ├── model.onnx
            │   └── test_data_set_0
            │       ├── input_0.pb
            │       ├── input_1.pb
            │       ├── output_0.pb
            │       └── output_1.pb

        Args:
            repro_dir: The directory to export the repro files to. Defaults to current
                working directory if None.
            name: An optional name for the test case folder: "test_{name}".

        Returns:
            The path to the exported repro directory.
        """
        if repro_dir is None:
            repro_dir = os.getcwd()
        repro_dir = os.path.join(repro_dir, 'onnx_debug')
        onnx_graph, onnx_params_dict = _onnx_graph_from_aten_graph(self.graph, self.export_options, self.params_dict)
        proto, _ = _onnx_proto_from_onnx_graph(onnx_graph, self.export_options, onnx_params_dict)
        return OnnxTestCaseRepro.create_test_case_repro(proto, self.input_args, self.pt_outs, repro_dir, name)

    @_beartype.beartype
    def _graph_partition_pivot(self) -> int:
        """Find the pivot index to partition the graph.

        The pivot is the node that splits the graph into two parts. Each part should
        have the similar amount of nodes, excluding non essential ops, defined in
        `_EXCLUDED_NODE_KINDS`, such as `prim::Constant`.
        If the graph has an odd number of nodes, the upper part will have one more node.
        If the graph does not have any node that can be partitioned, return -1.

        Returns:
            The index of the pivot node.
        """
        included_node_indices = [i for i, n in enumerate(self.graph.nodes()) if n.kind() not in self._EXCLUDED_NODE_KINDS]
        half_idx = len(included_node_indices) // 2 - 1
        if half_idx >= 0 and len(included_node_indices) > half_idx:
            return included_node_indices[half_idx] + 1
        return -1

    @_beartype.beartype
    def _partition_upper_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())

        def _process_bridge_value_for_upper(new_outputs: List[torch.Value], bridge_value: torch.Value) -> torch.Value:
            new_outputs.append(bridge_value)
            return bridge_value
        new_outputs: List[torch.Value] = []
        process_bridge_value_for_upper = functools.partial(_process_bridge_value_for_upper, new_outputs)
        _, dropped_nodes, complete_upper_nodes_set, _ = self._partition_nodes(graph, pivot, process_bridge_value_for_upper)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)
        for node in reversed(dropped_nodes):
            node.destroy()
        for i, input in reversed(list(enumerate(list(graph.inputs())))):
            if not _has_uses_by_nodes(input, complete_upper_nodes_set) and input not in new_outputs:
                try:
                    graph.eraseInput(i)
                except RuntimeError as e:
                    print(input, graph)
                    raise e
        return graph

    @_beartype.beartype
    def _partition_lower_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())
        original_inputs = list(graph.inputs())
        new_outputs = []

        def _process_bridge_value_for_lower(graph: torch.Graph, bridge_value: torch.Value) -> torch.Value:
            new_input = graph.addInput()
            bridge_value.replaceAllUsesWith(new_input)
            new_input.copyMetadata(bridge_value)
            return new_input
        process_bridge_value_for_lower = functools.partial(_process_bridge_value_for_lower, graph)
        upper_nodes, lower_nodes, _, complete_lower_nodes_set = self._partition_nodes(graph, pivot, process_bridge_value_for_lower)
        for output in original_outputs:
            if _produced_by(output, lower_nodes):
                new_outputs.append(output)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)
        for input in original_inputs:
            if _has_uses_by_nodes(input, complete_lower_nodes_set):
                new_input = graph.addInput()
                input.replaceAllUsesWith(new_input)
                new_input.copyMetadata(input)
        for node in reversed(upper_nodes):
            if node not in complete_lower_nodes_set:
                try:
                    node.destroy()
                except RuntimeError as e:
                    print(node, graph)
                    raise e
        for _ in original_inputs:
            graph.eraseInput(0)
        return graph

    @_beartype.beartype
    def _partition_node(self, node: torch.Node, complete_upper_nodes_set: Set[torch.Node], complete_lower_nodes_set: Set[torch.Node], original_graph_outputs: Set[torch.Value], covered_bridge_values: Set[torch.Value], process_bridge_value: Callable[[torch.Value], torch.Value]):
        if node in complete_lower_nodes_set:
            return
        if _node_has_uses_by(node, complete_lower_nodes_set) and node.kind() in self._EXCLUDED_NODE_KINDS:
            complete_lower_nodes_set.update(_all_nodes([node]))
            for input in node.inputs():
                if input in covered_bridge_values:
                    continue
                self._partition_node(input.node(), complete_upper_nodes_set, complete_lower_nodes_set, original_graph_outputs, covered_bridge_values, process_bridge_value)
        else:
            for output in node.outputs():
                if output in covered_bridge_values:
                    continue
                if _has_uses_by_nodes(output, complete_lower_nodes_set) or output in original_graph_outputs:
                    covered_bridge_values.add(process_bridge_value(output))

    @_beartype.beartype
    def _partition_nodes(self, graph: torch.Graph, pivot: int, process_bridge_value: Callable[[torch.Value], torch.Value]) -> Tuple[List[torch.Node], List[torch.Node], Set[torch.Node], Set[torch.Node]]:
        nodes = list(graph.nodes())
        upper_nodes = nodes[:pivot]
        lower_nodes = nodes[pivot:]
        complete_upper_nodes_set = _all_nodes(upper_nodes)
        complete_lower_nodes_set = _all_nodes(lower_nodes)
        original_graph_outputs = set(graph.outputs())
        covered_bridge_values = set(graph.inputs())
        for node in upper_nodes:
            self._partition_node(node, complete_upper_nodes_set, complete_lower_nodes_set, original_graph_outputs, covered_bridge_values, process_bridge_value)
        return (upper_nodes, lower_nodes, complete_upper_nodes_set, complete_lower_nodes_set)

    @_beartype.beartype
    def _bridge_kwargs(self):
        pt_outs = self.pt_outs
        graph_outputs = list(self.graph.outputs())
        assert pt_outs is not None
        assert len(graph_outputs) == len(pt_outs), f'{len(graph_outputs)} vs {len(pt_outs)}\nGraph: {self.graph}'
        return {v.debugName(): o for v, o in zip(graph_outputs, pt_outs)}

    @_beartype.beartype
    def _args_and_params_for_partition_graph(self, graph: torch.Graph, bridge_kwargs: Mapping[str, Union[_NumericType, Sequence[_NumericType]]], full_kwargs: Mapping[str, torch.Tensor], full_params: Mapping[str, torch.Tensor]):
        input_names = [input.debugName() for input in graph.inputs()]
        args = tuple((bridge_kwargs[k] for k in input_names if k in bridge_kwargs))
        args += tuple((full_kwargs[k] for k in input_names if k in full_kwargs))
        params = {k: full_params[k] for k in input_names if k in full_params}
        assert len(args) + len(params) == len(input_names), f'{len(args)} + {len(params)} vs {len(input_names)}: {input_names}'
        return (args, params)

    @_beartype.beartype
    def verify_export(self, options: VerificationOptions) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
        """
        Verify the export from TorchScript IR graph to ONNX.

        Export the TorchScript IR graph to ONNX, with the inputs, parameters and export
        options recorded in this object. Then verify the exported ONNX graph against
        the original TorchScript IR graph under the provided verification options.

        Args:
            options: The verification options.

        Returns:
            error: The AssertionError raised during the verification. Returns None if no
            error is raised.
            onnx_graph: The exported ONNX graph in TorchScript IR format.
            onnx_outs: The outputs from running exported ONNX model under the onnx
            backend in `options`.
            pt_outs: The outputs from running the TorchScript IR graph.
        """
        return verify_aten_graph(self.graph, input_args=self.input_args, params_dict=self.params_dict, export_options=self.export_options, verification_options=options)

    @_beartype.beartype
    def find_mismatch(self, options: Optional[VerificationOptions]=None):
        """
        Find all mismatches between the TorchScript IR graph and the exported onnx model.

        Binary searches the model graph to find the minimal subgraph that exhibits the
        mismatch. A `GraphInfo` object is created for each subgraph, recording the test
        inputs and export options, as well as the validation results.

        Args:
            options: The verification options.
        """
        self.clear()
        if options is None:
            options = VerificationOptions()
        if self.export_options.verbose:
            print(self.graph)
        if len(list(self.graph.outputs())) == 0:
            return
        assert len(self.input_args) + len(self.params_dict) == len(list(self.graph.inputs())), f'Number of graph inputs({len(list(self.graph.inputs()))}) does not match the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)}).'
        self.mismatch_error, self._onnx_graph, self.pt_outs, _ = self.verify_export(options)
        if self.mismatch_error is None:
            return
        if self.essential_node_count() <= 1:
            return
        full_kwargs = {k.debugName(): v for k, v in zip(self.graph.inputs(), self.input_args)}
        full_params = self.params_dict
        upper_graph = self._partition_upper_graph()
        upper_args, upper_params = self._args_and_params_for_partition_graph(upper_graph, {}, full_kwargs, full_params)
        self.upper_graph_info = GraphInfo(upper_graph, upper_args, upper_params, self.export_options, id=self.id + '0')
        self.upper_graph_info.find_mismatch(options)
        bridge_kwargs = self.upper_graph_info._bridge_kwargs()
        lower_graph = self._partition_lower_graph()
        lower_args, lower_params = self._args_and_params_for_partition_graph(lower_graph, bridge_kwargs, full_kwargs, full_params)
        self.lower_graph_info = GraphInfo(lower_graph, lower_args, lower_params, self.export_options, id=self.id + '1')
        self.lower_graph_info.find_mismatch(options)