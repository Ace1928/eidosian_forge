import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
def _assert_valueinfo_shape(self, onnx_model: ModelProto, value_infos: List[ValueInfoProto]) -> None:
    """Assert onnx_model.value_info should be the same as expected value_infos
        Instead of exact symbol, use -1 to represent symbolic shape in expected value_infos
        """
    for expected_vi in value_infos:
        shape = self._get_shape_from_name(onnx_model, expected_vi.name)
        assert shape is not None, f'{onnx_model}'
        if expected_vi.type.HasField('tensor_type'):
            expected_shape = expected_vi.type.tensor_type.shape
        elif expected_vi.type.HasField('sparse_tensor_type'):
            expected_shape = expected_vi.type.sparse_tensor_type.shape
        assert len(shape.dim) == len(expected_shape.dim), f'{onnx_model}'
        for dim_i, dim in enumerate(shape.dim):
            expected_dim = expected_shape.dim[dim_i]
            if expected_dim.dim_value == -1:
                assert dim.dim_param, f'{onnx_model}'
            else:
                assert dim.dim_value == expected_dim.dim_value, f'{onnx_model}'