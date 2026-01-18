import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def _genStackUsage(self):
    for x in range(5000):
        yield defer.succeed(1)
    returnValue(0)