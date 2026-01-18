import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def _assert_equal_shape_strides(arr1, arr2):
    self.assertEqual(arr1.shape, arr2.shape)
    self.assertEqual(arr1.strides, arr2.strides)