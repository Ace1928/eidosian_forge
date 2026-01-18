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
class TestExternalDataToArray(unittest.TestCase):
    serialization_format: str = 'protobuf'

    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name
        self._model_file_path: str = os.path.join(self.temp_dir, 'model.onnx')
        self.large_data = np.random.rand(10, 60, 100).astype(np.float32)
        self.small_data = (200, 300)
        self.model = self.create_test_model()

    @property
    def model_file_path(self):
        return self._model_file_path

    def tearDown(self) -> None:
        self._temp_dir_obj.cleanup()

    def create_test_model(self) -> ModelProto:
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, self.large_data.shape)
        input_init = helper.make_tensor(name='X', data_type=TensorProto.FLOAT, dims=self.large_data.shape, vals=self.large_data.tobytes(), raw=True)
        shape_data = np.array(self.small_data, np.int64)
        shape_init = helper.make_tensor(name='Shape', data_type=TensorProto.INT64, dims=shape_data.shape, vals=shape_data.tobytes(), raw=True)
        C = helper.make_tensor_value_info('C', TensorProto.INT64, self.small_data)
        reshape = onnx.helper.make_node('Reshape', inputs=['X', 'Shape'], outputs=['Y'])
        cast = onnx.helper.make_node('Cast', inputs=['Y'], outputs=['C'], to=TensorProto.INT64)
        graph_def = helper.make_graph([reshape, cast], 'test-model', [X], [C], initializer=[input_init, shape_init])
        model = helper.make_model(graph_def, producer_name='onnx-example')
        return model

    @unittest.skipIf(serialization_format != 'protobuf', 'check_model supports protobuf only when provided as a path')
    def test_check_model(self) -> None:
        checker.check_model(self.model)

    def test_reshape_inference_with_external_data_fail(self) -> None:
        onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
        model_without_external_data = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
        self.assertRaises(shape_inference.InferenceError, shape_inference.infer_shapes, model_without_external_data, strict_mode=True)

    def test_to_array_with_external_data(self) -> None:
        onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
        model = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
        loaded_large_data = to_array(model.graph.initializer[0], self.temp_dir)
        np.testing.assert_allclose(loaded_large_data, self.large_data)

    def test_save_model_with_external_data_multiple_times(self) -> None:
        onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, location=None, size_threshold=1024, convert_attribute=True)
        model_without_loading_external = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        self.assertTrue(large_input_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(large_input_tensor, self.temp_dir), self.large_data)
        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(not small_shape_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(small_shape_tensor), self.small_data)
        onnx.save_model(model_without_loading_external, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, location=None, size_threshold=0, convert_attribute=True)
        model_without_loading_external = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        self.assertTrue(large_input_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(large_input_tensor, self.temp_dir), self.large_data)
        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(small_shape_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(small_shape_tensor, self.temp_dir), self.small_data)