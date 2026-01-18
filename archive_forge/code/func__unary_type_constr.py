from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def _unary_type_constr(blah):
    self.assertEqual(blah, expected_blah[0])
    expected_blah[0] = None
    return ct.float32