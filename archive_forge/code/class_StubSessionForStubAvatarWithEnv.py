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
@implementer(session.ISessionSetEnv)
class StubSessionForStubAvatarWithEnv(StubSessionForStubAvatar):
    """
    Same as StubSessionForStubAvatar, but supporting environment variables
    setting.

    End users would want to have the same class annotated with
    C{@implementer(session.ISession, session.ISessionSetEnv)}. The interfaces
    are split for backwards compatibility, so we split it here to test
    this compatibility too.

    @ivar environ: a L{dict} of environment variables passed to the setEnv
    method.
    """

    def __init__(self, avatar):
        super().__init__(avatar)
        self.environ = {}
        self.environAtPty = {}

    def setEnv(self, name, value):
        """
        If the requested environment variable is 'FAIL', fail.  If it is
        'IGNORED', raise EnvironmentVariableNotPermitted, which should cause
        it to be silently ignored.  Otherwise, store the requested
        environment variable.

        (Real applications should normally implement an allowed list rather
        than a blocked list.)
        """
        if name == b'FAIL':
            raise RuntimeError('disallowed environment variable name')
        elif name == b'IGNORED':
            raise session.EnvironmentVariableNotPermitted('ignored environment variable name')
        else:
            self.environ[name] = value

    def getPty(self, term, windowSize, modes):
        """
        Just a simple implementation which records the current environment
        when PTY is requested.
        """
        self.environAtPty = self.environ.copy()