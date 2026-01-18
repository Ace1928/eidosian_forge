import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import (
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase
class WorkerLogObserverTests(TestCase):
    """
    Tests for L{WorkerLogObserver}.
    """

    def test_emit(self):
        """
        L{WorkerLogObserver} forwards data to L{managercommands.TestWrite}.
        """
        calls = []

        class FakeClient:

            def callRemote(self, method, **kwargs):
                calls.append((method, kwargs))
        observer = WorkerLogObserver(FakeClient())
        observer.emit({'message': ['Some log']})
        self.assertEqual(calls, [(managercommands.TestWrite, {'out': 'Some log'})])