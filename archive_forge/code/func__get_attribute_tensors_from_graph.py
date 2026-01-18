import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def _get_attribute_tensors_from_graph(onnx_model_proto_graph: GraphProto) -> Iterable[TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model graph."""
    for node in onnx_model_proto_graph.node:
        for attribute in node.attribute:
            if attribute.HasField('t'):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(attribute, _get_attribute_tensors_from_graph)