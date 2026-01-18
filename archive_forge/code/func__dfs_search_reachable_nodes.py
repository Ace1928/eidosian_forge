from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def _dfs_search_reachable_nodes(self, node_output_name: str, graph_input_names: list[str], reachable_nodes: list[NodeProto]) -> None:
    if node_output_name in graph_input_names:
        return
    for node in self.graph.node:
        if node_output_name not in node.output:
            continue
        if node in reachable_nodes:
            continue
        reachable_nodes.append(node)
        for name in node.input:
            self._dfs_search_reachable_nodes(name, graph_input_names, reachable_nodes)