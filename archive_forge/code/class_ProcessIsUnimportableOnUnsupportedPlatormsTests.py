import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
class ProcessIsUnimportableOnUnsupportedPlatormsTests(TestCase):
    """
    Tests to ensure that L{twisted.internet.process} is unimportable on
    platforms where it does not work (namely Windows).
    """

    @skipIf(not platform.isWindows(), 'Only relevant on Windows.')
    def test_unimportableOnWindows(self):
        """
        L{twisted.internet.process} is unimportable on Windows.
        """
        with self.assertRaises(ImportError):
            import twisted.internet.process
            twisted.internet.process