from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
class TestNotAllowToLoadExternalDataOutsideModelDirectory(TestLoadExternalDataBase):
    """Essential test to check that onnx (validate) C++ code will not allow to load external_data outside the model
    directory.
    """

    def create_external_data_tensor(self, value: list[Any], tensor_name: str, location: str='') -> TensorProto:
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = location or f'{tensor_name}.bin'
        set_external_data(tensor, location=tensor_filename)
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def test_check_model(self) -> None:
        """We only test the model validation as onnxruntime uses this to load the model."""
        self.model_filename = self.create_test_model('../../file.bin')
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_relative(self) -> None:
        """More relative path test."""
        self.model_filename = self.create_test_model('../test/../file.bin')
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_absolute(self) -> None:
        """ONNX checker disallows using absolute path as location in external tensor."""
        self.model_filename = self.create_test_model('//file.bin')
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)