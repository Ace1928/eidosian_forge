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
class Ender(ProcessProtocol):

    def childDataReceived(self, fd, data):
        msg('childDataReceived(%d, %r)' % (fd, data))
        self.transport.loseConnection()

    def childConnectionLost(self, childFD):
        msg('childConnectionLost(%d)' % (childFD,))
        lost.append(childFD)

    def processExited(self, reason):
        msg(f'processExited({reason!r})')

    def processEnded(self, reason):
        msg(f'processEnded({reason!r})')
        ended.callback([reason])