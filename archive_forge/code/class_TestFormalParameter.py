import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
class TestFormalParameter(unittest.TestCase):

    def test_init(self):
        name = 'input1'
        type_str = 'tensor(float)'
        description = 'The first input.'
        param_option = defs.OpSchema.FormalParameterOption.Single
        is_homogeneous = True
        min_arity = 1
        differentiation_category = defs.OpSchema.DifferentiationCategory.Unknown
        formal_parameter = defs.OpSchema.FormalParameter(name, type_str, description, param_option=param_option, is_homogeneous=is_homogeneous, min_arity=min_arity, differentiation_category=differentiation_category)
        self.assertEqual(formal_parameter.name, name)
        self.assertEqual(formal_parameter.type_str, type_str)
        self.assertEqual(formal_parameter.description, description)
        self.assertEqual(formal_parameter.option, param_option)
        self.assertEqual(formal_parameter.is_homogeneous, is_homogeneous)
        self.assertEqual(formal_parameter.min_arity, min_arity)
        self.assertEqual(formal_parameter.differentiation_category, differentiation_category)