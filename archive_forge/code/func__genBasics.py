import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def _genBasics(self):
    x = (yield getThing())
    self.assertEqual(x, 'hi')
    try:
        yield getOwie()
    except ZeroDivisionError as e:
        self.assertEqual(str(e), 'OMG')
    returnValue('WOOSH')