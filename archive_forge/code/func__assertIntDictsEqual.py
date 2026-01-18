import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def _assertIntDictsEqual(self, dict1, dict2):
    self.assertEqual(len(dict1), len(dict1), 'resulting dictionary is of different size')
    for key in dict1.keys():
        self.assertEqual(int(dict1[key]), int(dict2[key]))