from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
class TestDataShapeParserDims(unittest.TestCase):

    def setUp(self):
        self.sym = datashape.TypeSymbolTable()

    def test_fixed_dims(self):
        self.assertEqual(parse('3 * bool', self.sym), ct.DataShape(ct.Fixed(3), ct.bool_))
        self.assertEqual(parse('7 * 3 * bool', self.sym), ct.DataShape(ct.Fixed(7), ct.Fixed(3), ct.bool_))
        self.assertEqual(parse('5 * 3 * 12 * bool', self.sym), ct.DataShape(ct.Fixed(5), ct.Fixed(3), ct.Fixed(12), ct.bool_))
        self.assertEqual(parse('2 * 3 * 4 * 5 * bool', self.sym), ct.DataShape(ct.Fixed(2), ct.Fixed(3), ct.Fixed(4), ct.Fixed(5), ct.bool_))

    def test_typevar_dims(self):
        self.assertEqual(parse('M * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.bool_))
        self.assertEqual(parse('A * B * bool', self.sym), ct.DataShape(ct.TypeVar('A'), ct.TypeVar('B'), ct.bool_))
        self.assertEqual(parse('A... * X * 3 * bool', self.sym), ct.DataShape(ct.Ellipsis(ct.TypeVar('A')), ct.TypeVar('X'), ct.Fixed(3), ct.bool_))

    def test_var_dims(self):
        self.assertEqual(parse('var * bool', self.sym), ct.DataShape(ct.Var(), ct.bool_))
        self.assertEqual(parse('var * var * bool', self.sym), ct.DataShape(ct.Var(), ct.Var(), ct.bool_))
        self.assertEqual(parse('M * 5 * var * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Fixed(5), ct.Var(), ct.bool_))

    def test_ellipses(self):
        self.assertEqual(parse('... * bool', self.sym), ct.DataShape(ct.Ellipsis(), ct.bool_))
        self.assertEqual(parse('M * ... * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(), ct.bool_))
        self.assertEqual(parse('M * ... * 3 * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(), ct.Fixed(3), ct.bool_))