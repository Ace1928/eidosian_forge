import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
@requires_traitsui
class TestDatetimeEditor(SimpleEditorTestMixin, unittest.TestCase):
    traitsui_name = 'DatetimeEditor'
    factory_name = 'datetime_editor'

    def test_str_to_obj_conversions(self):
        obj = None
        obj_str = _datetime_to_datetime_str(obj)
        self.assertEqual(obj_str, '')
        self.assertEqual(_datetime_str_to_datetime(obj_str), obj)
        obj = datetime.datetime(2019, 1, 13)
        obj_str = _datetime_to_datetime_str(obj)
        self.assertIsInstance(obj_str, str)
        self.assertEqual(_datetime_str_to_datetime(obj_str), obj)
        obj_str = '2020-02-15T11:12:13'
        obj = _datetime_str_to_datetime(obj_str)
        self.assertIsInstance(obj, datetime.datetime)
        self.assertEqual(_datetime_to_datetime_str(obj), obj_str)
        obj_str = ''
        obj = _datetime_str_to_datetime(obj_str)
        self.assertIsNone(obj)
        self.assertEqual(_datetime_to_datetime_str(obj), obj_str)