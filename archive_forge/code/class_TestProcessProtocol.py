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
class TestProcessProtocol(ProcessProtocol):
    """
            Process protocol captures own presence in
            process.reapProcessHandlers at time of .processEnded() callback.

            @ivar deferred: A deferred fired when the .processEnded() callback
                has completed.
            @type deferred: L{Deferred<defer.Deferred>}
            """

    def __init__(self):
        self.deferred = Deferred()

    def processEnded(self, status):
        """
                Capture whether the process has already been removed
                from process.reapProcessHandlers.

                @param status: unused
                """
        from twisted.internet import process
        handlers = process.reapProcessHandlers
        processes = handlers.values()
        if self.transport in processes:
            results.append('process present but should not be')
        else:
            results.append('process already removed as desired')
        self.deferred.callback(None)