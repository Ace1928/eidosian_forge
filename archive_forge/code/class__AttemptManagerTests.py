import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(platformType != 'posix', 'twisted.mail only works on posix')
class _AttemptManagerTests(TestCase):
    """
    Test the behavior of L{_AttemptManager}.

    @type tmpdir: L{bytes}
    @ivar tmpdir: The path to a temporary directory holding the message files.

    @type reactor: L{MemoryReactorClock}
    @ivar reactor: The reactor used for test purposes.

    @type eventLog: L{None} or L{dict} of L{bytes} -> L{object}
    @ivar eventLog: Information about the last informational log message
        generated or none if no log message has been generated.

    @type noisyAttemptMgr: L{_AttemptManager}
    @ivar noisyAttemptMgr: An attempt manager which generates informational
        log messages.

    @type quietAttemptMgr: L{_AttemptManager}
    @ivar quietAttemptMgr: An attempt manager which does not generate
        informational log messages.

    @type noisyMessage: L{bytes}
    @ivar noisyMessage: The full base pathname of the message to be used with
        the noisy attempt manager.

    @type quietMessage: L{bytes}
    @ivar quietMessage: The full base pathname of the message to be used with
        the quiet.
    """

    def setUp(self):
        """
        Set up a temporary directory for the queue, attempt managers with the
        noisy flag on and off, message files for use with each attempt manager,
        and a reactor.  Also, register to be notified when log messages are
        generated.
        """
        self.tmpdir = self.mktemp()
        os.mkdir(self.tmpdir)
        self.reactor = MemoryReactorClock()
        self.eventLog = None
        log.addObserver(self._logObserver)
        self.noisyAttemptMgr = _AttemptManager(DummySmartHostSMTPRelayingManager(DummyQueue(self.tmpdir)), True, self.reactor)
        self.quietAttemptMgr = _AttemptManager(DummySmartHostSMTPRelayingManager(DummyQueue(self.tmpdir)), False, self.reactor)
        noisyBaseName = 'noisyMessage'
        quietBaseName = 'quietMessage'
        self.noisyMessage = os.path.join(self.tmpdir, noisyBaseName)
        self.quietMessage = os.path.join(self.tmpdir, quietBaseName)
        open(self.noisyMessage + '-D', 'w').close()
        open(self.quietMessage + '-D', 'w').close()
        self.noisyAttemptMgr.manager.managed['noisyRelayer'] = [noisyBaseName]
        self.quietAttemptMgr.manager.managed['quietRelayer'] = [quietBaseName]
        with open(self.noisyMessage + '-H', 'wb') as envelope:
            pickle.dump(['from-noisy@domain', 'to-noisy@domain'], envelope)
        with open(self.quietMessage + '-H', 'wb') as envelope:
            pickle.dump(['from-quiet@domain', 'to-quiet@domain'], envelope)

    def tearDown(self):
        """
        Unregister for log events and remove the temporary directory.
        """
        log.removeObserver(self._logObserver)
        shutil.rmtree(self.tmpdir)

    def _logObserver(self, eventDict):
        """
        A log observer.

        @type eventDict: L{dict} of L{bytes} -> L{object}
        @param eventDict: Information about the last informational log message
            generated.
        """
        self.eventLog = eventDict

    def test_initNoisyDefault(self):
        """
        When an attempt manager is created without the noisy parameter, the
        noisy instance variable should default to true.
        """
        am = _AttemptManager(DummySmartHostSMTPRelayingManager(DummyQueue(self.tmpdir)))
        self.assertTrue(am.noisy)

    def test_initNoisy(self):
        """
        When an attempt manager is created with the noisy parameter set to
        true, the noisy instance variable should be set to true.
        """
        self.assertTrue(self.noisyAttemptMgr.noisy)

    def test_initQuiet(self):
        """
        When an attempt manager is created with the noisy parameter set to
        false, the noisy instance variable should be set to false.
        """
        self.assertFalse(self.quietAttemptMgr.noisy)

    def test_initReactorDefault(self):
        """
        When an attempt manager is created without the reactor parameter, the
        reactor instance variable should default to the global reactor.
        """
        am = _AttemptManager(DummySmartHostSMTPRelayingManager(DummyQueue(self.tmpdir)))
        self.assertEqual(am.reactor, reactor)

    def test_initReactor(self):
        """
        When an attempt manager is created with a reactor provided, the
        reactor instance variable should default to that reactor.
        """
        self.assertEqual(self.noisyAttemptMgr.reactor, self.reactor)

    def test_notifySuccessNoisy(self):
        """
        For an attempt manager with the noisy flag set, notifySuccess should
        result in a log message.
        """
        self.noisyAttemptMgr.notifySuccess('noisyRelayer', self.noisyMessage)
        self.assertTrue(self.eventLog)

    def test_notifySuccessQuiet(self):
        """
        For an attempt manager with the noisy flag not set, notifySuccess
        should result in no log message.
        """
        self.quietAttemptMgr.notifySuccess('quietRelayer', self.quietMessage)
        self.assertFalse(self.eventLog)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_notifyFailureNoisy(self):
        """
        For an attempt manager with the noisy flag set, notifyFailure should
        result in a log message.
        """
        self.noisyAttemptMgr.notifyFailure('noisyRelayer', self.noisyMessage)
        self.assertTrue(self.eventLog)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_notifyFailureQuiet(self):
        """
        For an attempt manager with the noisy flag not set, notifyFailure
        should result in no log message.
        """
        self.quietAttemptMgr.notifyFailure('quietRelayer', self.quietMessage)
        self.assertFalse(self.eventLog)

    def test_notifyDoneNoisy(self):
        """
        For an attempt manager with the noisy flag set, notifyDone should
        result in a log message.
        """
        self.noisyAttemptMgr.notifyDone('noisyRelayer')
        self.assertTrue(self.eventLog)

    def test_notifyDoneQuiet(self):
        """
        For an attempt manager with the noisy flag not set, notifyDone
        should result in no log message.
        """
        self.quietAttemptMgr.notifyDone('quietRelayer')
        self.assertFalse(self.eventLog)

    def test_notifyNoConnectionNoisy(self):
        """
        For an attempt manager with the noisy flag set, notifyNoConnection
        should result in a log message.
        """
        self.noisyAttemptMgr.notifyNoConnection('noisyRelayer')
        self.assertTrue(self.eventLog)
        self.reactor.advance(60)

    def test_notifyNoConnectionQuiet(self):
        """
        For an attempt manager with the noisy flag not set, notifyNoConnection
        should result in no log message.
        """
        self.quietAttemptMgr.notifyNoConnection('quietRelayer')
        self.assertFalse(self.eventLog)
        self.reactor.advance(60)