from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
class TestDataShapeParseBasicDType(unittest.TestCase):

    def setUp(self):
        self.sym = datashape.TypeSymbolTable()

    def test_bool(self):
        self.assertEqual(parse('bool', self.sym), ct.DataShape(ct.bool_))

    def test_signed_integers(self):
        self.assertEqual(parse('int8', self.sym), ct.DataShape(ct.int8))
        self.assertEqual(parse('int16', self.sym), ct.DataShape(ct.int16))
        self.assertEqual(parse('int32', self.sym), ct.DataShape(ct.int32))
        self.assertEqual(parse('int64', self.sym), ct.DataShape(ct.int64))
        self.assertEqual(parse('int', self.sym), ct.DataShape(ct.int_))
        self.assertEqual(parse('int', self.sym), parse('int32', self.sym))
        self.assertEqual(parse('intptr', self.sym), ct.DataShape(ct.intptr))

    def test_unsigned_integers(self):
        self.assertEqual(parse('uint8', self.sym), ct.DataShape(ct.uint8))
        self.assertEqual(parse('uint16', self.sym), ct.DataShape(ct.uint16))
        self.assertEqual(parse('uint32', self.sym), ct.DataShape(ct.uint32))
        self.assertEqual(parse('uint64', self.sym), ct.DataShape(ct.uint64))
        self.assertEqual(parse('uintptr', self.sym), ct.DataShape(ct.uintptr))

    def test_float(self):
        self.assertEqual(parse('float16', self.sym), ct.DataShape(ct.float16))
        self.assertEqual(parse('float32', self.sym), ct.DataShape(ct.float32))
        self.assertEqual(parse('float64', self.sym), ct.DataShape(ct.float64))
        self.assertEqual(parse('real', self.sym), ct.DataShape(ct.real))
        self.assertEqual(parse('real', self.sym), parse('float64', self.sym))

    def test_null(self):
        self.assertEqual(parse('null', self.sym), ct.DataShape(ct.null))

    def test_void(self):
        self.assertEqual(parse('void', self.sym), ct.DataShape(ct.void))

    def test_object(self):
        self.assertEqual(parse('object', self.sym), ct.DataShape(ct.object_))

    def test_complex(self):
        self.assertEqual(parse('complex[float32]', self.sym), ct.DataShape(ct.complex_float32))
        self.assertEqual(parse('complex[float64]', self.sym), ct.DataShape(ct.complex_float64))
        self.assertEqual(parse('complex', self.sym), ct.DataShape(ct.complex_))
        self.assertEqual(parse('complex', self.sym), parse('complex[float64]', self.sym))

    def test_option(self):
        self.assertEqual(parse('option[int32]', self.sym), ct.DataShape(ct.Option(ct.int32)))
        self.assertEqual(parse('?int32', self.sym), ct.DataShape(ct.Option(ct.int32)))
        self.assertEqual(parse('2 * 3 * option[int32]', self.sym), ct.DataShape(ct.Fixed(2), ct.Fixed(3), ct.Option(ct.int32)))
        self.assertEqual(parse('2 * 3 * ?int32', self.sym), ct.DataShape(ct.Fixed(2), ct.Fixed(3), ct.Option(ct.int32)))
        self.assertEqual(parse('2 * option[3 * int32]', self.sym), ct.DataShape(ct.Fixed(2), ct.Option(ct.DataShape(ct.Fixed(3), ct.int32))))
        self.assertEqual(parse('2 * ?3 * int32', self.sym), ct.DataShape(ct.Fixed(2), ct.Option(ct.DataShape(ct.Fixed(3), ct.int32))))

    def test_raise(self):
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, 'boot', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, 'int33', self.sym)