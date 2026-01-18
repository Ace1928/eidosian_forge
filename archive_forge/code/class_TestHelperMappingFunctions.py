import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
class TestHelperMappingFunctions(unittest.TestCase):

    @pytest.mark.filterwarnings('error::DeprecationWarning')
    def test_tensor_dtype_to_np_dtype_not_throw_warning(self) -> None:
        _ = helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT)

    @pytest.mark.filterwarnings('error::DeprecationWarning')
    def test_tensor_dtype_to_storage_tensor_dtype_not_throw_warning(self) -> None:
        _ = helper.tensor_dtype_to_storage_tensor_dtype(TensorProto.FLOAT)

    @pytest.mark.filterwarnings('error::DeprecationWarning')
    def test_tensor_dtype_to_field_not_throw_warning(self) -> None:
        _ = helper.tensor_dtype_to_field(TensorProto.FLOAT)

    @pytest.mark.filterwarnings('error::DeprecationWarning')
    def test_np_dtype_to_tensor_dtype_not_throw_warning(self) -> None:
        _ = helper.np_dtype_to_tensor_dtype(np.dtype('float32'))

    def test_tensor_dtype_to_np_dtype_bfloat16(self) -> None:
        self.assertEqual(helper.tensor_dtype_to_np_dtype(TensorProto.BFLOAT16), np.dtype('float32'))

    def test_tensor_dtype_to_storage_tensor_dtype_bfloat16(self) -> None:
        self.assertEqual(helper.tensor_dtype_to_storage_tensor_dtype(TensorProto.BFLOAT16), TensorProto.UINT16)

    def test_tensor_dtype_to_field_bfloat16(self) -> None:
        self.assertEqual(helper.tensor_dtype_to_field(TensorProto.BFLOAT16), 'int32_data')