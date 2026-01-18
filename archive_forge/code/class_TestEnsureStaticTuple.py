import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
class TestEnsureStaticTuple(tests.TestCase):

    def test_is_static_tuple(self):
        st = static_tuple.StaticTuple('foo')
        st2 = static_tuple.expect_static_tuple(st)
        self.assertIs(st, st2)

    def test_is_tuple(self):
        t = ('foo',)
        st = static_tuple.expect_static_tuple(t)
        self.assertIsInstance(st, static_tuple.StaticTuple)
        self.assertEqual(t, st)

    def test_flagged_is_static_tuple(self):
        debug.debug_flags.add('static_tuple')
        st = static_tuple.StaticTuple('foo')
        st2 = static_tuple.expect_static_tuple(st)
        self.assertIs(st, st2)

    def test_flagged_is_tuple(self):
        debug.debug_flags.add('static_tuple')
        t = ('foo',)
        self.assertRaises(TypeError, static_tuple.expect_static_tuple, t)