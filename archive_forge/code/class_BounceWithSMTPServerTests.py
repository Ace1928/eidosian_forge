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
class BounceWithSMTPServerTests(TestCase):
    """
    Tests for L{twisted.mail.mail.BounceDomain} with
    L{twisted.mail.smtp.SMTPServer}.
    """

    def test_rejected(self):
        """
        Incoming emails to a SMTP server with L{twisted.mail.mail.BounceDomain}
        are rejected.
        """
        service = mail.mail.MailService()
        domain = mail.mail.BounceDomain()
        service.addDomain(b'foo.com', domain)
        factory = mail.protocols.SMTPFactory(service)
        protocol = factory.buildProtocol(None)
        deliverer = mail.protocols.SMTPDomainDelivery(service, None, None)
        protocol.delivery = deliverer
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.lineReceived(b'HELO baz.net')
        protocol.lineReceived(b'MAIL FROM:<a@baz.net>')
        protocol.lineReceived(b'RCPT TO:<any@foo.com>')
        protocol.lineReceived(b'QUIT')
        self.assertTrue(transport.disconnecting)
        protocol.connectionLost(None)
        self.assertEqual(transport.value().strip().split(b'\r\n')[-2], b'550 Cannot receive for specified address')