import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
def _get_shape_from_name(self, onnx_model: ModelProto, name: str) -> Optional[TensorShapeProto]:
    """Get shape from tensor_type or sparse_tensor_type according to given name"""
    inputs = list(onnx_model.graph.input)
    outputs = list(onnx_model.graph.output)
    valueinfos = list(onnx_model.graph.value_info)
    for v in inputs + outputs + valueinfos:
        if v.name == name:
            if v.type.HasField('tensor_type'):
                return v.type.tensor_type.shape
            if v.type.HasField('sparse_tensor_type'):
                return v.type.sparse_tensor_type.shape
    return None