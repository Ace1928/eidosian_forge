import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
class FailureLoggingMixin:

    def test_silent(self):
        """
            Don't log any errors.
            """

    def test_single(self):
        """
            Log a single error.
            """
        log.err(makeFailure())

    def test_double(self):
        """
            Log two errors.
            """
        log.err(makeFailure())
        log.err(makeFailure())

    def test_singleThenFail(self):
        """
            Log a single error, then fail.
            """
        log.err(makeFailure())
        1 + None