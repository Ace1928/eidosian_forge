import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def _requireTransportSecurityOverSSLTest(self, capabilities):
    """
        Verify that when L{smtp.ESMTPClient} connects to a server over a
        transport providing L{ISSLTransport}, C{requireTransportSecurity} is
        C{True}, and it is presented with the given capabilities, it will try
        to send its mail and not first attempt to negotiate TLS using the
        I{STARTTLS} protocol action.

        @param capabilities: Bytes to include in the test server's capability
            response.  These must be formatted exactly as required by the
            protocol, including a line which ends the capability response.
        @type param: L{bytes}

        @raise: C{self.failureException} if the behavior of
            C{self.clientProtocol} is not as described.
        """
    transport = StringTransport()
    directlyProvides(transport, interfaces.ISSLTransport)
    self.clientProtocol.makeConnection(transport)
    self.clientProtocol.dataReceived(self.SERVER_GREETING)
    transport.clear()
    self.clientProtocol.dataReceived(self.EHLO_RESPONSE + capabilities)
    self.assertEqual(b'MAIL FROM:<test@example.org>\r\n', transport.value())