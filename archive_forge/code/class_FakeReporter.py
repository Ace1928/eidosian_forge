import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
class FakeReporter(reporter.Reporter):
    """
        Fake reporter that does *not* implement done() but *does* implement
        printErrors, separator, printSummary, stream, write and writeln
        without deprecations.
        """
    done = None
    separator = None
    stream = None

    def printErrors(self, *args):
        pass

    def printSummary(self, *args):
        pass

    def write(self, *args):
        pass

    def writeln(self, *args):
        pass