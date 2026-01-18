from __future__ import annotations
import abc
from typing import Any, ClassVar, Iterable
import numpy as np
from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node, make_tensor_type_proto, np_dtype_to_tensor_dtype
from onnx.numpy_helper import to_array, unpack_int4
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
class RefAttrName:
    """Implements a link between a parameter of a function and an attribute in node.

    Args:
        name: name of the input
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!r})'