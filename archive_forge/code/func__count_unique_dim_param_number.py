import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
def _count_unique_dim_param_number(self, onnx_model: ModelProto) -> int:
    """Return the total number of unique symbolic shape"""
    symbol_shape_set = set()
    inputs = list(onnx_model.graph.input)
    outputs = list(onnx_model.graph.output)
    valueinfos = list(onnx_model.graph.value_info)
    for v in inputs + outputs + valueinfos:
        for dim in v.type.tensor_type.shape.dim:
            if dim.dim_param:
                symbol_shape_set.add(dim.dim_param)
    return len(symbol_shape_set)