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
class SearchUpwardsTests(TestCase):

    def testSearchupwards(self):
        os.makedirs('searchupwards/a/b/c')
        open('searchupwards/foo.txt', 'w').close()
        open('searchupwards/a/foo.txt', 'w').close()
        open('searchupwards/a/b/c/foo.txt', 'w').close()
        os.mkdir('searchupwards/bar')
        os.mkdir('searchupwards/bam')
        os.mkdir('searchupwards/a/bar')
        os.mkdir('searchupwards/a/b/bam')
        actual = util.searchupwards('searchupwards/a/b/c', files=['foo.txt'], dirs=['bar', 'bam'])
        expected = os.path.abspath('searchupwards') + os.sep
        self.assertEqual(actual, expected)
        shutil.rmtree('searchupwards')
        actual = util.searchupwards('searchupwards/a/b/c', files=['foo.txt'], dirs=['bar', 'bam'])
        expected = None
        self.assertEqual(actual, expected)