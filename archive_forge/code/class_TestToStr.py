from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
class TestToStr(unittest.TestCase):

    def test_new_type_float(self):
        self.assertEqual(tostr(0.5), '0.5')
        self.assertIs(tostr.handlers[float], tostr.handlers[None])

    def test_new_type_int(self):
        self.assertEqual(tostr(0), '0')
        self.assertIs(tostr.handlers[int], tostr.handlers[None])

    def test_new_type_str(self):
        self.assertEqual(tostr(DerivedStr(1)), '1')
        self.assertIs(tostr.handlers[DerivedStr], tostr.handlers[str])

    def test_new_type_list(self):
        self.assertEqual(tostr(DerivedList([1, 2])), '[1, 2]')
        self.assertIs(tostr.handlers[DerivedList], tostr.handlers[list])

    def test_new_type_dict(self):
        self.assertEqual(tostr(DerivedDict({1: 2})), '{1: 2}')
        self.assertIs(tostr.handlers[DerivedDict], tostr.handlers[dict])

    def test_new_type_tuple(self):
        self.assertEqual(tostr(DerivedTuple([1, 2])), '(1, 2)')
        self.assertIs(tostr.handlers[DerivedTuple], tostr.handlers[tuple])

    def test_new_type_namedtuple(self):
        self.assertEqual(tostr(NamedTuple(1, 2)), 'NamedTuple(x=1, y=2)')
        self.assertIs(tostr.handlers[NamedTuple], tostr.handlers[None])