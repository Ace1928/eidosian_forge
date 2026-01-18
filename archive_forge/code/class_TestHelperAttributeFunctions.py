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
class TestHelperAttributeFunctions(unittest.TestCase):

    def test_attr_float(self) -> None:
        attr = helper.make_attribute('float', 1.0)
        self.assertEqual(attr.name, 'float')
        self.assertEqual(attr.f, 1.0)
        checker.check_attribute(attr)
        attr = helper.make_attribute('float', 10000000000.0)
        self.assertEqual(attr.name, 'float')
        self.assertEqual(attr.f, 10000000000.0)
        checker.check_attribute(attr)

    def test_attr_int(self) -> None:
        attr = helper.make_attribute('int', 3)
        self.assertEqual(attr.name, 'int')
        self.assertEqual(attr.i, 3)
        checker.check_attribute(attr)
        attr = helper.make_attribute('int', 5)
        self.assertEqual(attr.name, 'int')
        self.assertEqual(attr.i, 5)
        checker.check_attribute(attr)
        attr = helper.make_attribute('int', 961)
        self.assertEqual(attr.name, 'int')
        self.assertEqual(attr.i, 961)
        checker.check_attribute(attr)
        attr = helper.make_attribute('int', 5889)
        self.assertEqual(attr.name, 'int')
        self.assertEqual(attr.i, 5889)
        checker.check_attribute(attr)

    def test_attr_doc_string(self) -> None:
        attr = helper.make_attribute('a', 'value')
        self.assertEqual(attr.name, 'a')
        self.assertEqual(attr.doc_string, '')
        attr = helper.make_attribute('a', 'value', 'doc')
        self.assertEqual(attr.name, 'a')
        self.assertEqual(attr.doc_string, 'doc')

    def test_attr_string(self) -> None:
        attr = helper.make_attribute('str', b'test')
        self.assertEqual(attr.name, 'str')
        self.assertEqual(attr.s, b'test')
        checker.check_attribute(attr)
        attr = helper.make_attribute('str', 'test')
        self.assertEqual(attr.name, 'str')
        self.assertEqual(attr.s, b'test')
        checker.check_attribute(attr)
        attr = helper.make_attribute('str', 'test')
        self.assertEqual(attr.name, 'str')
        self.assertEqual(attr.s, b'test')
        checker.check_attribute(attr)
        attr = helper.make_attribute('str', '')
        self.assertEqual(attr.name, 'str')
        self.assertEqual(helper.get_attribute_value(attr), b'')
        checker.check_attribute(attr)

    def test_attr_repeated_float(self) -> None:
        attr = helper.make_attribute('floats', [1.0, 2.0])
        self.assertEqual(attr.name, 'floats')
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        checker.check_attribute(attr)

    def test_attr_repeated_int(self) -> None:
        attr = helper.make_attribute('ints', [1, 2])
        self.assertEqual(attr.name, 'ints')
        self.assertEqual(list(attr.ints), [1, 2])
        checker.check_attribute(attr)

    def test_attr_repeated_mixed_floats_and_ints(self) -> None:
        attr = helper.make_attribute('mixed', [1, 2, 3.0, 4.5])
        self.assertEqual(attr.name, 'mixed')
        self.assertEqual(list(attr.floats), [1.0, 2.0, 3.0, 4.5])
        checker.check_attribute(attr)

    def test_attr_repeated_str(self) -> None:
        attr = helper.make_attribute('strings', ['str1', 'str2'])
        self.assertEqual(attr.name, 'strings')
        self.assertEqual(list(attr.strings), [b'str1', b'str2'])
        checker.check_attribute(attr)

    def test_attr_repeated_tensor_proto(self) -> None:
        tensors = [helper.make_tensor(name='a', data_type=TensorProto.FLOAT, dims=(1,), vals=np.ones(1)), helper.make_tensor(name='b', data_type=TensorProto.FLOAT, dims=(1,), vals=np.ones(1))]
        attr = helper.make_attribute('tensors', tensors)
        self.assertEqual(attr.name, 'tensors')
        self.assertEqual(list(attr.tensors), tensors)
        checker.check_attribute(attr)

    def test_attr_sparse_tensor_proto(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = helper.make_tensor(name='sparse_values', data_type=TensorProto.FLOAT, dims=[len(sparse_values)], vals=np.array(sparse_values).astype(np.float32), raw=False)
        linear_indices = [2, 3, 5]
        indices_tensor = helper.make_tensor(name='indices', data_type=TensorProto.INT64, dims=[len(linear_indices)], vals=np.array(linear_indices).astype(np.int64), raw=False)
        sparse_tensor = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        attr = helper.make_attribute('sparse_attr', sparse_tensor)
        self.assertEqual(attr.name, 'sparse_attr')
        checker.check_sparse_tensor(helper.get_attribute_value(attr))
        checker.check_attribute(attr)

    def test_attr_sparse_tensor_repeated_protos(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = helper.make_tensor(name='sparse_values', data_type=TensorProto.FLOAT, dims=[len(sparse_values)], vals=np.array(sparse_values).astype(np.float32), raw=False)
        linear_indices = [2, 3, 5]
        indices_tensor = helper.make_tensor(name='indices', data_type=TensorProto.INT64, dims=[len(linear_indices)], vals=np.array(linear_indices).astype(np.int64), raw=False)
        sparse_tensor = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        repeated_sparse = [sparse_tensor, sparse_tensor]
        attr = helper.make_attribute('sparse_attrs', repeated_sparse)
        self.assertEqual(attr.name, 'sparse_attrs')
        checker.check_attribute(attr)
        for s in helper.get_attribute_value(attr):
            checker.check_sparse_tensor(s)

    def test_attr_repeated_graph_proto(self) -> None:
        graphs = [GraphProto(), GraphProto()]
        graphs[0].name = 'a'
        graphs[1].name = 'b'
        attr = helper.make_attribute('graphs', graphs)
        self.assertEqual(attr.name, 'graphs')
        self.assertEqual(list(attr.graphs), graphs)
        checker.check_attribute(attr)

    def test_attr_type_proto(self) -> None:
        type_proto = TypeProto()
        attr = helper.make_attribute('type_proto', type_proto)
        self.assertEqual(attr.name, 'type_proto')
        self.assertEqual(attr.tp, type_proto)
        self.assertEqual(attr.type, AttributeProto.TYPE_PROTO)
        types = [TypeProto(), TypeProto()]
        attr = helper.make_attribute('type_protos', types)
        self.assertEqual(attr.name, 'type_protos')
        self.assertEqual(list(attr.type_protos), types)
        self.assertEqual(attr.type, AttributeProto.TYPE_PROTOS)

    def test_attr_empty_list(self) -> None:
        attr = helper.make_attribute('empty', [], attr_type=AttributeProto.STRINGS)
        self.assertEqual(attr.type, AttributeProto.STRINGS)
        self.assertEqual(len(attr.strings), 0)
        self.assertRaises(ValueError, helper.make_attribute, 'empty', [])

    def test_attr_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Inferred attribute type 'FLOAT'"):
            helper.make_attribute('test', 6.4, attr_type=AttributeProto.STRING)

    def test_is_attr_legal(self) -> None:
        attr = AttributeProto()
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        attr = AttributeProto()
        attr.name = 'test'
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        attr = AttributeProto()
        attr.name = 'test'
        attr.f = 1.0
        attr.i = 2
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)

    def test_is_attr_legal_verbose(self) -> None:

        def _set(attr: AttributeProto, type_: AttributeProto.AttributeType, var: str, value: Any) -> None:
            setattr(attr, var, value)
            attr.type = type_

        def _extend(attr: AttributeProto, type_: AttributeProto.AttributeType, var: List[Any], value: Any) -> None:
            var.extend(value)
            attr.type = type_
        SET_ATTR = [lambda attr: _set(attr, AttributeProto.FLOAT, 'f', 1.0), lambda attr: _set(attr, AttributeProto.INT, 'i', 1), lambda attr: _set(attr, AttributeProto.STRING, 's', b'str'), lambda attr: _extend(attr, AttributeProto.FLOATS, attr.floats, [1.0, 2.0]), lambda attr: _extend(attr, AttributeProto.INTS, attr.ints, [1, 2]), lambda attr: _extend(attr, AttributeProto.STRINGS, attr.strings, [b'a', b'b'])]
        for _i in range(100):
            attr = AttributeProto()
            attr.name = 'test'
            random.choice(SET_ATTR)(attr)
            checker.check_attribute(attr)
        for _i in range(100):
            attr = AttributeProto()
            attr.name = 'test'
            for func in random.sample(SET_ATTR, 2):
                func(attr)
            self.assertRaises(checker.ValidationError, checker.check_attribute, attr)