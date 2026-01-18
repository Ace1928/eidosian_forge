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
class NameToLabelTests(TestCase):
    """
    Tests for L{nameToLabel}.
    """

    def test_nameToLabel(self):
        """
        Test the various kinds of inputs L{nameToLabel} supports.
        """
        nameData = [('f', 'F'), ('fo', 'Fo'), ('foo', 'Foo'), ('fooBar', 'Foo Bar'), ('fooBarBaz', 'Foo Bar Baz')]
        for inp, out in nameData:
            got = util.nameToLabel(inp)
            self.assertEqual(got, out, f'nameToLabel({inp!r}) == {got!r} != {out!r}')