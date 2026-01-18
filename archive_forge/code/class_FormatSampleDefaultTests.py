import re
import unittest
from oslo_config import types
class FormatSampleDefaultTests(unittest.TestCase):

    def test_string(self):
        t = types.String()
        self.assertEqual([' bar '], t.format_defaults('foo', sample_default=' bar '))

    def test_string_non_str(self):
        t = types.String()
        e = Exception('bar')
        self.assertEqual(['bar'], t.format_defaults('', sample_default=e))

    def test_string_non_str_spaces(self):
        t = types.String()
        e = Exception(' bar ')
        self.assertEqual(['" bar "'], t.format_defaults('', sample_default=e))

    def test_list_string(self):
        t = types.List(item_type=types.String())
        test_list = ['foo', Exception(' bar ')]
        self.assertEqual(['foo," bar "'], t.format_defaults('', sample_default=test_list))

    def test_list_no_type(self):
        t = types.List()
        test_list = ['foo', Exception(' bar ')]
        self.assertEqual(['foo," bar "'], t.format_defaults('', sample_default=test_list))

    def test_list_not_list(self):
        t = types.List()
        self.assertEqual(['foo'], t.format_defaults('', sample_default=Exception('foo')))