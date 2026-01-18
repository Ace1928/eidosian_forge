import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def _genHandledTerminalFailure(self):
    try:
        yield defer.fail(TerminalException('Handled Terminal Failure'))
    except TerminalException:
        pass