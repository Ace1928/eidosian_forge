import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
class TestTyping(TestCase):
    """
    Check typing of basic indexing operations
    """

    def test_layout(self):
        """
        Check an appropriate layout is inferred for the result of array
        indexing.
        """
        func = arraydecl.get_array_index_type
        cty = types.Array(types.float64, 3, 'C')
        fty = types.Array(types.float64, 3, 'F')
        aty = types.Array(types.float64, 3, 'A')
        indices = [((), True, True), ((ellipsis,), True, True), ((intp,), True, False), ((slice2_type,), True, False), ((intp, slice2_type), True, False), ((slice2_type, intp), False, False), ((slice2_type, slice2_type), False, False), ((intp, slice3_type), False, False), ((slice3_type,), False, False), ((ellipsis, intp), False, True), ((ellipsis, slice2_type), False, True), ((ellipsis, intp, slice2_type), False, False), ((ellipsis, slice2_type, intp), False, True), ((ellipsis, slice2_type, slice2_type), False, False), ((ellipsis, slice3_type), False, False), ((ellipsis, slice3_type, intp), False, False), ((intp, ellipsis, intp), False, False), ((slice2_type, ellipsis, slice2_type), False, False), ((intp, intp, slice2_type), True, False), ((intp, ellipsis, intp, slice2_type), True, False), ((slice2_type, intp, intp), False, True), ((slice2_type, intp, ellipsis, intp), False, True), ((intp, slice2_type, intp), False, False), ((slice3_type, intp, intp), False, False), ((intp, intp, slice3_type), False, False)]
        for index_tuple, keep_c, _ in indices:
            index = types.Tuple(index_tuple)
            r = func(cty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'C' if keep_c else 'A', index_tuple)
            self.assertFalse(r.advanced)
        for index_tuple, _, keep_f in indices:
            index = types.Tuple(index_tuple)
            r = func(fty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'F' if keep_f else 'A', index_tuple)
            self.assertFalse(r.advanced)
        for index_tuple, _, _ in indices:
            index = types.Tuple(index_tuple)
            r = func(aty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'A')
            self.assertFalse(r.advanced)