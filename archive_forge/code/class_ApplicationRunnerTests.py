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
class ApplicationRunnerTests(TestCase):
    """
    Non-platform-specific tests for the platform-specific ApplicationRunner.
    """

    def setUp(self):
        config = twistd.ServerOptions()
        self.serviceMaker = MockServiceMaker()
        config.loadedPlugins = {'test_command': self.serviceMaker}
        config.subOptions = object()
        config.subCommand = 'test_command'
        self.config = config

    def test_applicationRunnerGetsCorrectApplication(self):
        """
        Ensure that a twistd plugin gets used in appropriate ways: it
        is passed its Options instance, and the service it returns is
        added to the application.
        """
        arunner = CrippledApplicationRunner(self.config)
        arunner.run()
        self.assertIs(self.serviceMaker.options, self.config.subOptions, 'ServiceMaker.makeService needs to be passed the correct sub Command object.')
        self.assertIs(self.serviceMaker.service, service.IService(arunner.application).services[0], "ServiceMaker.makeService's result needs to be set as a child of the Application.")

    def test_preAndPostApplication(self):
        """
        Test thet preApplication and postApplication methods are
        called by ApplicationRunner.run() when appropriate.
        """
        s = TestApplicationRunner(self.config)
        s.run()
        self.assertFalse(s.hadApplicationPreApplication)
        self.assertTrue(s.hadApplicationPostApplication)
        self.assertTrue(s.hadApplicationLogObserver)
        self.assertEqual(s.order, ['pre', 'log', 'post'])

    def _applicationStartsWithConfiguredID(self, argv, uid, gid):
        """
        Assert that given a particular command line, an application is started
        as a particular UID/GID.

        @param argv: A list of strings giving the options to parse.
        @param uid: An integer giving the expected UID.
        @param gid: An integer giving the expected GID.
        """
        self.config.parseOptions(argv)
        events = []

        class FakeUnixApplicationRunner(twistd._SomeApplicationRunner):

            def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
                events.append('environment')

            def shedPrivileges(self, euid, uid, gid):
                events.append(('privileges', euid, uid, gid))

            def startReactor(self, reactor, oldstdout, oldstderr):
                events.append('reactor')

            def removePID(self, pidfile):
                pass

        @implementer(service.IService, service.IProcess)
        class FakeService:
            parent = None
            running = None
            name = None
            processName = None
            uid = None
            gid = None

            def setName(self, name):
                pass

            def setServiceParent(self, parent):
                pass

            def disownServiceParent(self):
                pass

            def privilegedStartService(self):
                events.append('privilegedStartService')

            def startService(self):
                events.append('startService')

            def stopService(self):
                pass
        application = FakeService()
        verifyObject(service.IService, application)
        verifyObject(service.IProcess, application)
        runner = FakeUnixApplicationRunner(self.config)
        runner.preApplication()
        runner.application = application
        runner.postApplication()
        self.assertEqual(events, ['environment', 'privilegedStartService', ('privileges', False, uid, gid), 'startService', 'reactor'])

    @skipIf(not getattr(os, 'setuid', None), 'Platform does not support --uid/--gid twistd options.')
    def test_applicationStartsWithConfiguredNumericIDs(self):
        """
        L{postApplication} should change the UID and GID to the values
        specified as numeric strings by the configuration after running
        L{service.IService.privilegedStartService} and before running
        L{service.IService.startService}.
        """
        uid = 1234
        gid = 4321
        self._applicationStartsWithConfiguredID(['--uid', str(uid), '--gid', str(gid)], uid, gid)

    @skipIf(not getattr(os, 'setuid', None), 'Platform does not support --uid/--gid twistd options.')
    def test_applicationStartsWithConfiguredNameIDs(self):
        """
        L{postApplication} should change the UID and GID to the values
        specified as user and group names by the configuration after running
        L{service.IService.privilegedStartService} and before running
        L{service.IService.startService}.
        """
        user = 'foo'
        uid = 1234
        group = 'bar'
        gid = 4321
        patchUserDatabase(self.patch, user, uid, group, gid)
        self._applicationStartsWithConfiguredID(['--uid', user, '--gid', group], uid, gid)

    def test_startReactorRunsTheReactor(self):
        """
        L{startReactor} calls L{reactor.run}.
        """
        reactor = DummyReactor()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertTrue(reactor.called, 'startReactor did not call reactor.run()')

    def test_applicationRunnerChoosesReactorIfNone(self):
        """
        L{ApplicationRunner} chooses a reactor if none is specified.
        """
        reactor = DummyReactor()
        self.patch(internet, 'reactor', reactor)
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(None, None, None)
        self.assertTrue(reactor.called)

    def test_applicationRunnerCapturesSignal(self):
        """
        If the reactor exits with a signal, the application runner caches
        the signal.
        """

        class DummyReactorWithSignal(ReactorBase):
            """
            A dummy reactor, providing a C{run} method, and setting the
            _exitSignal attribute to a nonzero value.
            """

            def installWaker(self):
                """
                Dummy method, does nothing.
                """

            def run(self):
                """
                A fake run method setting _exitSignal to a nonzero value
                """
                self._exitSignal = 2
        reactor = DummyReactorWithSignal()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertEquals(2, runner._exitSignal)

    def test_applicationRunnerIgnoresNoSignal(self):
        """
        The runner sets its _exitSignal instance attribute to None if
        the reactor does not implement L{_ISupportsExitSignalCapturing}.
        """

        class DummyReactorWithExitSignalAttribute:
            """
            A dummy reactor, providing a C{run} method, and setting the
            _exitSignal attribute to a nonzero value.
            """

            def installWaker(self):
                """
                Dummy method, does nothing.
                """

            def run(self):
                """
                A fake run method setting _exitSignal to a nonzero value
                that should be ignored.
                """
                self._exitSignal = 2
        reactor = DummyReactorWithExitSignalAttribute()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertEquals(None, runner._exitSignal)