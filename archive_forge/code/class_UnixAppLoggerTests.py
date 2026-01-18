import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixAppLoggerTests(TestCase):
    """
    Tests for L{UnixAppLogger}.

    @ivar signals: list of signal handlers installed.
    @type signals: C{list}
    """

    def setUp(self):
        """
        Fake C{signal.signal} for not installing the handlers but saving them
        in C{self.signals}.
        """
        self.signals = []

        def fakeSignal(sig, f):
            self.signals.append((sig, f))
        self.patch(signal, 'signal', fakeSignal)

    def test_getLogObserverStdout(self):
        """
        When non-daemonized and C{logfile} is empty or set to C{-},
        L{UnixAppLogger._getLogObserver} returns a log observer pointing at
        C{sys.stdout}.
        """
        logFiles = _patchTextFileLogObserver(self.patch)
        logger = UnixAppLogger({'logfile': '-', 'nodaemon': True})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 1)
        self.assertIs(logFiles[0], sys.stdout)
        logger = UnixAppLogger({'logfile': '', 'nodaemon': True})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 2)
        self.assertIs(logFiles[1], sys.stdout)

    def test_getLogObserverStdoutDaemon(self):
        """
        When daemonized and C{logfile} is set to C{-},
        L{UnixAppLogger._getLogObserver} raises C{SystemExit}.
        """
        logger = UnixAppLogger({'logfile': '-', 'nodaemon': False})
        error = self.assertRaises(SystemExit, logger._getLogObserver)
        self.assertEqual(str(error), 'Daemons cannot log to stdout, exiting!')

    def test_getLogObserverFile(self):
        """
        When C{logfile} contains a file name, L{app.AppLogger._getLogObserver}
        returns a log observer pointing at the specified path, and a signal
        handler rotating the log is installed.
        """
        logFiles = _patchTextFileLogObserver(self.patch)
        filename = self.mktemp()
        sut = UnixAppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath(filename))
        self.assertEqual(len(self.signals), 1)
        self.assertEqual(self.signals[0][0], signal.SIGUSR1)
        d = Deferred()

        def rotate():
            d.callback(None)
        logFiles[0].rotate = rotate
        rotateLog = self.signals[0][1]
        rotateLog(None, None)
        return d

    def test_getLogObserverDontOverrideSignalHandler(self):
        """
        If a signal handler is already installed,
        L{UnixAppLogger._getLogObserver} doesn't override it.
        """

        def fakeGetSignal(sig):
            self.assertEqual(sig, signal.SIGUSR1)
            return object()
        self.patch(signal, 'getsignal', fakeGetSignal)
        filename = self.mktemp()
        sut = UnixAppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(self.signals, [])

    def test_getLogObserverDefaultFile(self):
        """
        When daemonized and C{logfile} is empty, the observer returned by
        L{UnixAppLogger._getLogObserver} points at C{twistd.log} in the current
        directory.
        """
        logFiles = _patchTextFileLogObserver(self.patch)
        logger = UnixAppLogger({'logfile': '', 'nodaemon': False})
        observer = logger._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath('twistd.log'))

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_getLogObserverSyslog(self):
        """
        If C{syslog} is set to C{True}, L{UnixAppLogger._getLogObserver} starts
        a L{syslog.SyslogObserver} with given C{prefix}.
        """
        logs = _setupSyslog(self)
        logger = UnixAppLogger({'syslog': True, 'prefix': 'test-prefix'})
        observer = logger._getLogObserver()
        self.assertEqual(logs, ['test-prefix'])
        observer({'a': 'b'})
        self.assertEqual(logs, ['test-prefix', {'a': 'b'}])