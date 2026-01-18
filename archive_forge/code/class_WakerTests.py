import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class WakerTests(WarningCheckerTestCase):

    def test_noWakerConstructionWarnings(self):
        """
        No warnings are generated when constructing the waker.
        """
        waker = _Waker()
        warnings = self.flushWarnings()
        self.assertEqual(len(warnings), 0, warnings)
        waker.connectionLost(None)
        warnings = self.flushWarnings()
        self.assertEqual(len(warnings), 0, warnings)