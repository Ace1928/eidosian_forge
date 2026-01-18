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
class AppLoggerTests(TestCase):
    """
    Tests for L{app.AppLogger}.

    @ivar observers: list of observers installed during the tests.
    @type observers: C{list}
    """

    def setUp(self):
        """
        Override L{globaLogBeginner.beginLoggingTo} so that we can trace the
        observers installed in C{self.observers}.
        """
        self.observers = []

        def beginLoggingTo(observers):
            for observer in observers:
                self.observers.append(observer)
                globalLogPublisher.addObserver(observer)
        self.patch(globalLogBeginner, 'beginLoggingTo', beginLoggingTo)

    def tearDown(self):
        """
        Remove all installed observers.
        """
        for observer in self.observers:
            globalLogPublisher.removeObserver(observer)

    def _makeObserver(self):
        """
        Make a new observer which captures all logs sent to it.

        @return: An observer that stores all logs sent to it.
        @rtype: Callable that implements L{ILogObserver}.
        """

        @implementer(ILogObserver)
        class TestObserver:
            _logs = []

            def __call__(self, event):
                self._logs.append(event)
        return TestObserver()

    def _checkObserver(self, observer):
        """
        Ensure that initial C{twistd} logs are written to logs.

        @param observer: The observer made by L{self._makeObserver).
        """
        self.assertEqual(self.observers, [observer])
        self.assertIn('starting up', observer._logs[0]['log_format'])
        self.assertIn('reactor class', observer._logs[1]['log_format'])

    def test_start(self):
        """
        L{app.AppLogger.start} calls L{globalLogBeginner.addObserver}, and then
        writes some messages about twistd and the reactor.
        """
        logger = app.AppLogger({})
        observer = self._makeObserver()
        logger._getLogObserver = lambda: observer
        logger.start(Componentized())
        self._checkObserver(observer)

    def test_startUsesApplicationLogObserver(self):
        """
        When the L{ILogObserver} component is available on the application,
        that object will be used as the log observer instead of constructing a
        new one.
        """
        application = Componentized()
        observer = self._makeObserver()
        application.setComponent(ILogObserver, observer)
        logger = app.AppLogger({})
        logger.start(application)
        self._checkObserver(observer)

    def _setupConfiguredLogger(self, application, extraLogArgs={}, appLogger=app.AppLogger):
        """
        Set up an AppLogger which exercises the C{logger} configuration option.

        @type application: L{Componentized}
        @param application: The L{Application} object to pass to
            L{app.AppLogger.start}.
        @type extraLogArgs: C{dict}
        @param extraLogArgs: extra values to pass to AppLogger.
        @type appLogger: L{AppLogger} class, or a subclass
        @param appLogger: factory for L{AppLogger} instances.

        @rtype: C{list}
        @return: The logs accumulated by the log observer.
        """
        observer = self._makeObserver()
        logArgs = {'logger': lambda: observer}
        logArgs.update(extraLogArgs)
        logger = appLogger(logArgs)
        logger.start(application)
        return observer

    def test_startUsesConfiguredLogObserver(self):
        """
        When the C{logger} key is specified in the configuration dictionary
        (i.e., when C{--logger} is passed to twistd), the initial log observer
        will be the log observer returned from the callable which the value
        refers to in FQPN form.
        """
        application = Componentized()
        self._checkObserver(self._setupConfiguredLogger(application))

    def test_configuredLogObserverBeatsComponent(self):
        """
        C{--logger} takes precedence over a L{ILogObserver} component set on
        Application.
        """
        observer = self._makeObserver()
        application = Componentized()
        application.setComponent(ILogObserver, observer)
        self._checkObserver(self._setupConfiguredLogger(application))
        self.assertEqual(observer._logs, [])

    def test_configuredLogObserverBeatsLegacyComponent(self):
        """
        C{--logger} takes precedence over a L{LegacyILogObserver} component
        set on Application.
        """
        nonlogs = []
        application = Componentized()
        application.setComponent(LegacyILogObserver, nonlogs.append)
        self._checkObserver(self._setupConfiguredLogger(application))
        self.assertEqual(nonlogs, [])

    def test_loggerComponentBeatsLegacyLoggerComponent(self):
        """
        A L{ILogObserver} takes precedence over a L{LegacyILogObserver}
        component set on Application.
        """
        nonlogs = []
        observer = self._makeObserver()
        application = Componentized()
        application.setComponent(ILogObserver, observer)
        application.setComponent(LegacyILogObserver, nonlogs.append)
        logger = app.AppLogger({})
        logger.start(application)
        self._checkObserver(observer)
        self.assertEqual(nonlogs, [])

    @skipIf(not _twistd_unix, 'twistd unix not available')
    @skipIf(not syslog, 'syslog not available')
    def test_configuredLogObserverBeatsSyslog(self):
        """
        C{--logger} takes precedence over a C{--syslog} command line
        argument.
        """
        logs = _setupSyslog(self)
        application = Componentized()
        self._checkObserver(self._setupConfiguredLogger(application, {'syslog': True}, UnixAppLogger))
        self.assertEqual(logs, [])

    def test_configuredLogObserverBeatsLogfile(self):
        """
        C{--logger} takes precedence over a C{--logfile} command line
        argument.
        """
        application = Componentized()
        path = self.mktemp()
        self._checkObserver(self._setupConfiguredLogger(application, {'logfile': 'path'}))
        self.assertFalse(os.path.exists(path))

    def test_getLogObserverStdout(self):
        """
        When logfile is empty or set to C{-}, L{app.AppLogger._getLogObserver}
        returns a log observer pointing at C{sys.stdout}.
        """
        logger = app.AppLogger({'logfile': '-'})
        logFiles = _patchTextFileLogObserver(self.patch)
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 1)
        self.assertIs(logFiles[0], sys.stdout)
        logger = app.AppLogger({'logfile': ''})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 2)
        self.assertIs(logFiles[1], sys.stdout)

    def test_getLogObserverFile(self):
        """
        When passing the C{logfile} option, L{app.AppLogger._getLogObserver}
        returns a log observer pointing at the specified path.
        """
        logFiles = _patchTextFileLogObserver(self.patch)
        filename = self.mktemp()
        sut = app.AppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath(filename))

    def test_stop(self):
        """
        L{app.AppLogger.stop} removes the observer created in C{start}, and
        reinitialize its C{_observer} so that if C{stop} is called several
        times it doesn't break.
        """
        removed = []
        observer = object()

        def remove(observer):
            removed.append(observer)
        self.patch(globalLogPublisher, 'removeObserver', remove)
        logger = app.AppLogger({})
        logger._observer = observer
        logger.stop()
        self.assertEqual(removed, [observer])
        logger.stop()
        self.assertEqual(removed, [observer])
        self.assertIsNone(logger._observer)

    def test_legacyObservers(self):
        """
        L{app.AppLogger} using a legacy logger observer still works, wrapping
        it in a compat shim.
        """
        logs = []
        logger = app.AppLogger({})

        @implementer(LegacyILogObserver)
        class LoggerObserver:
            """
            An observer which implements the legacy L{LegacyILogObserver}.
            """

            def __call__(self, x):
                """
                Add C{x} to the logs list.
                """
                logs.append(x)
        logger._observerFactory = lambda: LoggerObserver()
        logger.start(Componentized())
        self.assertIn('starting up', textFromEventDict(logs[0]))
        warnings = self.flushWarnings([self.test_legacyObservers])
        self.assertEqual(len(warnings), 0, warnings)

    def test_unmarkedObserversDeprecated(self):
        """
        L{app.AppLogger} using a logger observer which does not implement
        L{ILogObserver} or L{LegacyILogObserver} will be wrapped in a compat
        shim and raise a L{DeprecationWarning}.
        """
        logs = []
        logger = app.AppLogger({})
        logger._getLogObserver = lambda: logs.append
        logger.start(Componentized())
        self.assertIn('starting up', textFromEventDict(logs[0]))
        warnings = self.flushWarnings([self.test_unmarkedObserversDeprecated])
        self.assertEqual(warnings[0]['message'], 'Passing a logger factory which makes log observers which do not implement twisted.logger.ILogObserver or twisted.python.log.ILogObserver to twisted.application.app.AppLogger was deprecated in Twisted 16.2. Please use a factory that produces twisted.logger.ILogObserver (or the legacy twisted.python.log.ILogObserver) implementing objects instead.')
        self.assertEqual(len(warnings), 1, warnings)