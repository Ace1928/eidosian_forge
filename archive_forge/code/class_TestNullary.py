from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestNullary(TestCase):

    def test_repr(self):

        def foo():
            pass
        wrapped = Nullary(foo)
        self.assertEqual(repr(wrapped), repr(foo))

    def test_called_with_arguments(self):
        l = []

        def foo(*args, **kwargs):
            l.append((args, kwargs))
        wrapped = Nullary(foo, 1, 2, a='b')
        wrapped()
        self.assertEqual(l, [((1, 2), {'a': 'b'})])

    def test_returns_wrapped(self):
        ret = object()
        wrapped = Nullary(lambda: ret)
        self.assertIs(ret, wrapped())

    def test_raises(self):
        wrapped = Nullary(lambda: 1 / 0)
        self.assertRaises(ZeroDivisionError, wrapped)