import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class test_base2(unittest.TestCase):

    def indexed_assign(self, list, index, value):
        list[index] = value

    def indexed_range_assign(self, list, index1, index2, value):
        list[index1:index2] = value

    def extended_slice_assign(self, list, index1, index2, step, value):
        list[index1:index2:step] = value

    def check_values(self, name, default_value, good_values, bad_values, actual_values=None, mapped_values=None):
        obj = self.obj
        value = default_value
        self.assertEqual(getattr(obj, name), value)
        if actual_values is None:
            actual_values = good_values
        i = 0
        for value in good_values:
            setattr(obj, name, value)
            self.assertEqual(getattr(obj, name), actual_values[i])
            if mapped_values is not None:
                self.assertEqual(getattr(obj, name + '_'), mapped_values[i])
            i += 1
        for value in bad_values:
            self.assertRaises(TraitError, setattr, obj, name, value)