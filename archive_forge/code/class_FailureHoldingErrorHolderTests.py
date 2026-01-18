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
class FailureHoldingErrorHolderTests(ErrorHolderTestsMixin, TestHolderTests):
    """
    Tests for L{runner.ErrorHolder} behaving similarly to L{runner.TestHolder}
    when constructed with a L{Failure} representing its error.
    """

    def setUp(self):
        self.description = 'description'
        try:
            raise self.exceptionForTests
        except ZeroDivisionError:
            self.error = failure.Failure()
        self.holder = runner.ErrorHolder(self.description, self.error)
        self.result = self.TestResultStub()