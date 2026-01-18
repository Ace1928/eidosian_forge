import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class SessionWithNoAvatarTests(RegistryUsingMixin, TestCase):
    """
    Test for the SSHSession interface.  Several of the methods (request_shell,
    request_exec, request_pty_req, request_env, request_window_change) would
    create a 'session' instance variable from the avatar if one didn't exist
    when they were called.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def setUp(self):
        RegistryUsingMixin.setUp(self)
        components.registerAdapter(StubSessionForStubAvatar, StubAvatar, session.ISession)
        self.session = session.SSHSession()
        self.session.avatar = StubAvatar()
        self.assertIsNone(self.session.session)

    def assertSessionProvidesISession(self):
        """
        self.session.session should provide I{ISession}.
        """
        self.assertTrue(session.ISession.providedBy(self.session.session), 'ISession not provided by %r' % self.session.session)

    def test_requestShellGetsSession(self):
        """
        If an ISession adapter isn't already present, request_shell should get
        one.
        """
        self.session.requestReceived(b'shell', b'')
        self.assertSessionProvidesISession()

    def test_requestExecGetsSession(self):
        """
        If an ISession adapter isn't already present, request_exec should get
        one.
        """
        self.session.requestReceived(b'exec', common.NS(b'success'))
        self.assertSessionProvidesISession()

    def test_requestPtyReqGetsSession(self):
        """
        If an ISession adapter isn't already present, request_pty_req should
        get one.
        """
        self.session.requestReceived(b'pty_req', session.packRequest_pty_req(b'term', (0, 0, 0, 0), b''))
        self.assertSessionProvidesISession()

    def test_requestWindowChangeGetsSession(self):
        """
        If an ISession adapter isn't already present, request_window_change
        should get one.
        """
        self.session.requestReceived(b'window_change', session.packRequest_window_change((1, 1, 1, 1)))
        self.assertSessionProvidesISession()