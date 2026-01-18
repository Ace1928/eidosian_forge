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
@parameterized.parameterized_class([{'serialization_format': 'protobuf'}, {'serialization_format': 'textproto'}])
class TestLoadExternalData(TestLoadExternalDataBase):

    def test_load_external_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_load_external_data_for_model(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format, load_external_data=False)
        load_external_data_for_model(model, self.temp_dir)
        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_external_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)
        temp_dir = os.path.join(self.temp_dir, 'save_copy')
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(model, new_model_filename, self.serialization_format)
        new_model = onnx.load_model(new_model_filename, self.serialization_format)
        initializer_tensor = new_model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = new_model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)