import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional
import pydot
from onnx import GraphProto, ModelProto, NodeProto
def GetOpNodeProducer(embed_docstring: bool=False, **kwargs: Any) -> _NodeProducer:

    def really_get_op_node(op: NodeProto, op_id: int) -> pydot.Node:
        if op.name:
            node_name = f'{op.name}/{op.op_type} (op#{op_id})'
        else:
            node_name = f'{op.op_type} (op#{op_id})'
        for i, input_ in enumerate(op.input):
            node_name += '\n input' + str(i) + ' ' + input_
        for i, output in enumerate(op.output):
            node_name += '\n output' + str(i) + ' ' + output
        node = pydot.Node(node_name, **kwargs)
        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node
    return really_get_op_node