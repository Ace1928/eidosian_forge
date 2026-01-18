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
class TestHelperOptionalAndSequenceFunctions(unittest.TestCase):

    def test_make_optional(self) -> None:
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(5,), vals=values)
        optional = helper.make_optional(name='test', elem_type=OptionalProto.TENSOR, value=values_tensor)
        self.assertEqual(optional.name, 'test')
        self.assertEqual(optional.elem_type, OptionalProto.TENSOR)
        self.assertEqual(optional.tensor_value, values_tensor)
        values_sequence = helper.make_sequence(name='test', elem_type=SequenceProto.TENSOR, values=[values_tensor, values_tensor])
        optional = helper.make_optional(name='test', elem_type=OptionalProto.SEQUENCE, value=values_sequence)
        self.assertEqual(optional.name, 'test')
        self.assertEqual(optional.elem_type, OptionalProto.SEQUENCE)
        self.assertEqual(optional.sequence_value, values_sequence)
        optional_none = helper.make_optional(name='test', elem_type=OptionalProto.UNDEFINED, value=None)
        self.assertEqual(optional_none.name, 'test')
        self.assertEqual(optional_none.elem_type, OptionalProto.UNDEFINED)
        self.assertFalse(optional_none.HasField('tensor_value'))

    def test_make_optional_value_info(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(elem_type=2, shape=[5])
        tensor_val_into = helper.make_value_info(name='test', type_proto=tensor_type_proto)
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        optional_val_info = helper.make_value_info(name='test', type_proto=optional_type_proto)
        self.assertEqual(optional_val_info.name, 'test')
        self.assertTrue(optional_val_info.type.optional_type)
        self.assertEqual(optional_val_info.type.optional_type.elem_type, tensor_val_into.type)
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        optional_val_info = helper.make_value_info(name='test', type_proto=optional_type_proto)
        self.assertEqual(optional_val_info.name, 'test')
        self.assertTrue(optional_val_info.type.optional_type)
        sequence_value_info = helper.make_value_info(name='test', type_proto=tensor_type_proto)
        self.assertEqual(optional_val_info.type.optional_type.elem_type.sequence_type.elem_type, sequence_value_info.type)

    def test_make_seuence_value_info(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(elem_type=2, shape=None)
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = helper.make_value_info(name='test', type_proto=sequence_type_proto)
        sequence_val_info_prim = helper.make_tensor_sequence_value_info(name='test', elem_type=2, shape=None)
        self.assertEqual(sequence_val_info, sequence_val_info_prim)