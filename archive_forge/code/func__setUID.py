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