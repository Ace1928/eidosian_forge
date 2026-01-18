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
class TestSaveAllTensorsAsExternalData(unittest.TestCase):
    serialization_format: str = 'protobuf'

    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model = self.create_test_model_proto()

    def get_temp_model_filename(self):
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_data_tensors(self, tensors_data: list[tuple[list[Any], Any]]) -> list[TensorProto]:
        tensors = []
        for value, tensor_name in tensors_data:
            tensor = from_array(np.array(value))
            tensor.name = tensor_name
            tensors.append(tensor)
        return tensors

    def create_test_model_proto(self) -> ModelProto:
        tensors = self.create_data_tensors([(self.attribute_value, 'attribute_value'), (self.initializer_value, 'input_value')])
        constant_node = onnx.helper.make_node('Constant', inputs=[], outputs=['values'], value=tensors[0])
        inputs = [helper.make_tensor_value_info('input_value', onnx.TensorProto.FLOAT, self.initializer_value.shape)]
        graph = helper.make_graph([constant_node], 'test_graph', inputs=inputs, outputs=[], initializer=[tensors[1]])
        return helper.make_model(graph)

    @unittest.skipIf(serialization_format != 'protobuf', 'check_model supports protobuf only when provided as a path')
    def test_check_model(self) -> None:
        checker.check_model(self.model)

    def test_convert_model_to_external_data_with_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=1024)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField('data_location'))

    def test_convert_model_to_external_data_without_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

    def test_convert_model_to_external_data_from_one_file_with_location(self) -> None:
        model_file_path = self.get_temp_model_filename()
        external_data_file = str(uuid.uuid4())
        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True, location=external_data_file)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, external_data_file)))
        model = onnx.load_model(model_file_path, self.serialization_format)
        convert_model_from_external_data(model)
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(len(initializer_tensor.external_data))
        self.assertEqual(initializer_tensor.data_location, TensorProto.DEFAULT)
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(len(attribute_tensor.external_data))
        self.assertEqual(attribute_tensor.data_location, TensorProto.DEFAULT)
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_convert_model_to_external_data_from_one_file_without_location_uses_model_name(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, model_file_path)))

    def test_convert_model_to_external_data_one_file_per_tensor_without_attribute(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=False, convert_attribute=False)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'input_value')))
        self.assertFalse(os.path.isfile(os.path.join(self.temp_dir, 'attribute_value')))

    def test_convert_model_to_external_data_one_file_per_tensor_with_attribute(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=False, convert_attribute=True)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'input_value')))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'attribute_value')))

    def test_convert_model_to_external_data_does_not_convert_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0, convert_attribute=False, all_tensors_to_one_file=False)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'input_value')))
        self.assertFalse(os.path.isfile(os.path.join(self.temp_dir, 'attribute_value')))
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField('data_location'))
        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField('data_location'))

    def test_convert_model_to_external_data_converts_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0, convert_attribute=True)
        onnx.save_model(self.model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        self.assertTrue(initializer_tensor.HasField('data_location'))
        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)
        self.assertTrue(attribute_tensor.HasField('data_location'))

    def test_save_model_does_not_convert_to_external_data_and_saves_the_model(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=False)
        self.assertTrue(os.path.isfile(model_file_path))
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField('data_location'))
        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField('data_location'))

    def test_save_model_does_convert_and_saves_the_model(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=True, location=None, size_threshold=0, convert_attribute=False)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_model_without_loading_external_data(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=True, location=None, size_threshold=0, convert_attribute=False)
        model = onnx.load_model(model_file_path, self.serialization_format, load_external_data=False)
        onnx.save_model(model, model_file_path, self.serialization_format, save_as_external_data=True, location=None, size_threshold=0, convert_attribute=False)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField('data_location'))
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_model_with_existing_raw_data_should_override(self) -> None:
        model_file_path = self.get_temp_model_filename()
        original_raw_data = self.model.graph.initializer[0].raw_data
        onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=True, size_threshold=0)
        self.assertTrue(os.path.isfile(model_file_path))
        model = onnx.load_model(model_file_path, self.serialization_format, load_external_data=False)
        initializer_tensor = model.graph.initializer[0]
        initializer_tensor.raw_data = b'dummpy_raw_data'
        load_external_data_for_tensor(initializer_tensor, self.temp_dir)
        self.assertEqual(initializer_tensor.raw_data, original_raw_data)