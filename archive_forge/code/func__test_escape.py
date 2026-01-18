from __future__ import absolute_import
import unittest
import Cython.Compiler.PyrexTypes as PT
def _test_escape(self, func_name, test_data=TEST_DATA):
    escape = getattr(PT, func_name)
    for declaration, expected in test_data:
        escaped_value = escape(declaration)
        self.assertEqual(escaped_value, expected, "%s('%s') == '%s' != '%s'" % (func_name, declaration, escaped_value, expected))
        self.assertLessEqual(len(escaped_value), 64)