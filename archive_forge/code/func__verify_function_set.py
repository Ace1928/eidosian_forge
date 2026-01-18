import unittest
import onnx
from onnx import checker, utils
def _verify_function_set(self, extracted_model, function_set, func_domain):
    checker.check_model(extracted_model)
    self.assertEqual(len(extracted_model.functions), len(function_set))
    for function in function_set:
        self.assertIsNotNone(next((f for f in extracted_model.functions if f.name == function and f.domain == func_domain), None))