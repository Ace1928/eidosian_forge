import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional
import pydot
from onnx import GraphProto, ModelProto, NodeProto
def GetPydotGraph(graph: GraphProto, name: Optional[str]=None, rankdir: str='LR', node_producer: Optional[_NodeProducer]=None, embed_docstring: bool=False) -> pydot.Dot:
    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes: Dict[str, pydot.Node] = {}
    pydot_node_counts: Dict[str, int] = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(_escape_label(input_name + str(pydot_node_counts[input_name])), label=_escape_label(input_name), **BLOB_STYLE)
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(_escape_label(output_name + str(pydot_node_counts[output_name])), label=_escape_label(output_name), **BLOB_STYLE)
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph