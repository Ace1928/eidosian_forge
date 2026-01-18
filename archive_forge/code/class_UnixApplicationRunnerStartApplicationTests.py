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
class UnixApplicationRunnerStartApplicationTests(TestCase):
    """
    Tests for L{UnixApplicationRunner.startApplication}.
    """

    def test_setupEnvironment(self):
        """
        L{UnixApplicationRunner.startApplication} calls
        L{UnixApplicationRunner.setupEnvironment} with the chroot, rundir,
        nodaemon, umask, and pidfile parameters from the configuration it is
        constructed with.
        """
        options = twistd.ServerOptions()
        options.parseOptions(['--nodaemon', '--umask', '0070', '--chroot', '/foo/chroot', '--rundir', '/foo/rundir', '--pidfile', '/foo/pidfile'])
        application = service.Application('test_setupEnvironment')
        self.runner = UnixApplicationRunner(options)
        args = []

        def fakeSetupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
            args.extend((chroot, rundir, nodaemon, umask, pidfile))
        setupEnvironmentParameters = inspect.signature(self.runner.setupEnvironment).parameters
        fakeSetupEnvironmentParameters = inspect.signature(fakeSetupEnvironment).parameters
        fakeSetupEnvironmentParameters = fakeSetupEnvironmentParameters.copy()
        fakeSetupEnvironmentParameters.pop('self')
        self.assertEqual(setupEnvironmentParameters, fakeSetupEnvironmentParameters)
        self.patch(UnixApplicationRunner, 'setupEnvironment', fakeSetupEnvironment)
        self.patch(UnixApplicationRunner, 'shedPrivileges', lambda *a, **kw: None)
        self.patch(app, 'startApplication', lambda *a, **kw: None)
        self.runner.startApplication(application)
        self.assertEqual(args, ['/foo/chroot', '/foo/rundir', True, 56, '/foo/pidfile'])

    def test_shedPrivileges(self):
        """
        L{UnixApplicationRunner.shedPrivileges} switches the user ID
        of the process.
        """

        def switchUIDPass(uid, gid, euid):
            self.assertEqual(uid, 200)
            self.assertEqual(gid, 54)
            self.assertEqual(euid, 35)
        self.patch(_twistd_unix, 'switchUID', switchUIDPass)
        runner = UnixApplicationRunner({})
        runner.shedPrivileges(35, 200, 54)

    def test_shedPrivilegesError(self):
        """
        An unexpected L{OSError} when calling
        L{twisted.scripts._twistd_unix.shedPrivileges}
        terminates the process via L{SystemExit}.
        """

        def switchUIDFail(uid, gid, euid):
            raise OSError(errno.EBADF, 'fake')
        runner = UnixApplicationRunner({})
        self.patch(_twistd_unix, 'switchUID', switchUIDFail)
        exc = self.assertRaises(SystemExit, runner.shedPrivileges, 35, 200, None)
        self.assertEqual(exc.code, 1)

    def _setUID(self, wantedUser, wantedUid, wantedGroup, wantedGid, pidFile):
        """
        Common code for tests which try to pass the the UID to
        L{UnixApplicationRunner}.
        """
        patchUserDatabase(self.patch, wantedUser, wantedUid, wantedGroup, wantedGid)

        def initgroups(uid, gid):
            self.assertEqual(uid, wantedUid)
            self.assertEqual(gid, wantedGid)

        def setuid(uid):
            self.assertEqual(uid, wantedUid)

        def setgid(gid):
            self.assertEqual(gid, wantedGid)
        self.patch(util, 'initgroups', initgroups)
        self.patch(os, 'setuid', setuid)
        self.patch(os, 'setgid', setgid)
        options = twistd.ServerOptions()
        options.parseOptions(['--nodaemon', '--uid', str(wantedUid), '--pidfile', pidFile])
        application = service.Application('test_setupEnvironment')
        self.runner = UnixApplicationRunner(options)
        runner = UnixApplicationRunner(options)
        runner.startApplication(application)

    def test_setUidWithoutGid(self):
        """
        Starting an application with L{UnixApplicationRunner} configured
        with a UID and no GUID will result in the GUID being
        set to the default GUID for that UID.
        """
        self._setUID('foo', 5151, 'bar', 4242, self.mktemp() + '_test_setUidWithoutGid.pid')

    def test_setUidSameAsCurrentUid(self):
        """
        If the specified UID is the same as the current UID of the process,
        then a warning is displayed.
        """
        currentUid = os.getuid()
        self._setUID('morefoo', currentUid, 'morebar', 4343, 'test_setUidSameAsCurrentUid.pid')
        warningsShown = self.flushWarnings()
        expectedWarning = 'tried to drop privileges and setuid {} but uid is already {}; should we be root? Continuing.'.format(currentUid, currentUid)
        self.assertEqual(expectedWarning, warningsShown[0]['message'])
        self.assertEqual(1, len(warningsShown), warningsShown)