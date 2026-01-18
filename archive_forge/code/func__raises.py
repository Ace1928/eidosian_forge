import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
@inlineCallbacks
def _raises():
    try:
        yield _returns()
        raise TerminalException('boom normal return')
    except TerminalException:
        return traceback.format_exc()