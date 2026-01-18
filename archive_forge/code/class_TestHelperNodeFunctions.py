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
class TestHelperNodeFunctions(unittest.TestCase):

    def test_node_no_arg(self) -> None:
        self.assertTrue(defs.has('Relu'))
        node_def = helper.make_node('Relu', ['X'], ['Y'], name='test')
        self.assertEqual(node_def.op_type, 'Relu')
        self.assertEqual(node_def.name, 'test')
        self.assertEqual(list(node_def.input), ['X'])
        self.assertEqual(list(node_def.output), ['Y'])

    def test_attr_doc_string(self) -> None:
        node_def = helper.make_node('Relu', ['X'], ['Y'], name='test', doc_string='doc')
        self.assertEqual(node_def.doc_string, 'doc')

    def test_node_with_arg(self) -> None:
        self.assertTrue(defs.has('Relu'))
        node_def = helper.make_node('Relu', ['X'], ['Y'], arg_value=1)
        self.assertEqual(node_def.op_type, 'Relu')
        self.assertEqual(list(node_def.input), ['X'])
        self.assertEqual(list(node_def.output), ['Y'])
        self.assertEqual(len(node_def.attribute), 1)
        self.assertEqual(node_def.attribute[0], helper.make_attribute('arg_value', 1))

    def test_node_domain(self) -> None:
        node_def = helper.make_node('Relu', ['X'], ['Y'], name='test', doc_string='doc', domain='test.domain')
        self.assertEqual(node_def.domain, 'test.domain')

    def test_graph(self) -> None:
        node_def1 = helper.make_node('Relu', ['X'], ['Y'])
        node_def2 = helper.make_node('Add', ['X', 'Y'], ['Z'])
        value_info = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])]
        graph = helper.make_graph([node_def1, node_def2], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])], doc_string=None, value_info=value_info)
        self.assertEqual(graph.name, 'test')
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0], node_def1)
        self.assertEqual(graph.node[1], node_def2)
        self.assertEqual(graph.doc_string, '')
        self.assertEqual(graph.value_info[0], value_info[0])

    def test_graph_docstring(self) -> None:
        graph = helper.make_graph([], 'my graph', [], [], None, 'my docs')
        self.assertEqual(graph.name, 'my graph')
        self.assertEqual(graph.doc_string, 'my docs')

    def test_model(self) -> None:
        node_def = helper.make_node('Relu', ['X'], ['Y'])
        graph_def = helper.make_graph([node_def], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
        self.assertRaises(AttributeError, helper.make_model, graph_def, xxx=1)
        model_def = helper.make_model(graph_def, producer_name='test')
        self.assertEqual(model_def.producer_name, 'test')

    def test_model_docstring(self) -> None:
        graph = helper.make_graph([], 'my graph', [], [])
        model_def = helper.make_model(graph, doc_string='test')
        self.assertFalse(hasattr(model_def, 'name'))
        self.assertEqual(model_def.doc_string, 'test')

    def test_model_metadata_props(self) -> None:
        graph = helper.make_graph([], 'my graph', [], [])
        model_def = helper.make_model(graph, doc_string='test')
        helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
        checker.check_model(model_def)
        helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
        checker.check_model(model_def)
        dupe = model_def.metadata_props.add()
        dupe.key = 'Title'
        dupe.value = 'Other'
        self.assertRaises(checker.ValidationError, checker.check_model, model_def)

    def test_model_irversion(self) -> None:

        def mk_model(opset_versions: List[Tuple[str, int]]) -> ModelProto:
            graph = helper.make_graph([], 'my graph', [], [])
            return helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid(*pair) for pair in opset_versions])

        def test(opset_versions: List[Tuple[str, int]], ir_version: int) -> None:
            model = mk_model(opset_versions)
            self.assertEqual(model.ir_version, ir_version)
        test([('', 9)], 4)
        test([('', 10)], 5)
        test([('', 11)], 6)
        test([('', 12)], 7)
        test([('', 13)], 7)
        test([('', 14)], 7)
        test([('', 15)], 8)
        test([('', 16)], 8)
        test([('', 17)], 8)
        test([('', 18)], 8)
        test([('', 19)], 9)
        test([('', 20)], 9)
        test([('', 21)], 10)
        test([('ai.onnx', 9)], 4)
        test([('ai.onnx.ml', 2)], 6)
        test([('ai.onnx.ml', 3)], 8)
        test([('ai.onnx.ml', 4)], 9)
        test([('ai.onnx.ml', 5)], 10)
        test([('ai.onnx.training', 1)], 7)
        test([('', 10), ('ai.onnx.ml', 2)], 6)
        self.assertRaises(ValueError, mk_model, [('', 100)])