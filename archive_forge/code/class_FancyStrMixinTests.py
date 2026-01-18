import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
class FancyStrMixinTests(TestCase):
    """
    Tests for L{util.FancyStrMixin}.
    """

    def test_sequenceOfStrings(self):
        """
        If C{showAttributes} is set to a sequence of strings, C{__str__}
        renders using those by looking them up as attributes on the object.
        """

        class Foo(util.FancyStrMixin):
            showAttributes = ('first', 'second')
            first = 1
            second = 'hello'
        self.assertEqual(str(Foo()), "<Foo first=1 second='hello'>")

    def test_formatter(self):
        """
        If C{showAttributes} has an item that is a 2-tuple, C{__str__} renders
        the first item in the tuple as a key and the result of calling the
        second item with the value of the attribute named by the first item as
        the value.
        """

        class Foo(util.FancyStrMixin):
            showAttributes = ('first', ('second', lambda value: repr(value[::-1])))
            first = 'hello'
            second = 'world'
        self.assertEqual("<Foo first='hello' second='dlrow'>", str(Foo()))

    def test_override(self):
        """
        If C{showAttributes} has an item that is a 3-tuple, C{__str__} renders
        the second item in the tuple as a key, and the contents of the
        attribute named in the first item are rendered as the value. The value
        is formatted using the third item in the tuple.
        """

        class Foo(util.FancyStrMixin):
            showAttributes = ('first', ('second', '2nd', '%.1f'))
            first = 1
            second = 2.111
        self.assertEqual(str(Foo()), '<Foo first=1 2nd=2.1>')

    def test_fancybasename(self):
        """
        If C{fancybasename} is present, C{__str__} uses it instead of the class name.
        """

        class Foo(util.FancyStrMixin):
            fancybasename = 'Bar'
        self.assertEqual(str(Foo()), '<Bar>')

    def test_repr(self):
        """
        C{__repr__} outputs the same content as C{__str__}.
        """

        class Foo(util.FancyStrMixin):
            showAttributes = ('first', 'second')
            first = 1
            second = 'hello'
        obj = Foo()
        self.assertEqual(str(obj), repr(obj))