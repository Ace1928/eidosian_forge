from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def _binary_type_constr(a, b):
    self.assertEqual(a, expected_arg[0])
    self.assertEqual(b, expected_arg[1])
    expected_arg[0] = None
    expected_arg[1] = None
    return ct.float32