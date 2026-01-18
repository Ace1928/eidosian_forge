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
def _stringificationTest(self, stringifier):
    """
        Assert that the class name of a L{mail.mail.DomainWithDefaultDict}
        instance and the string-formatted underlying domain dictionary both
        appear in the string produced by the given string-returning function.

        @type stringifier: one-argument callable
        @param stringifier: either C{str} or C{repr}, to be used to get a
            string to make assertions against.
        """
    domain = mail.mail.DomainWithDefaultDict({}, 'Default')
    self.assertIn(domain.__class__.__name__, stringifier(domain))
    domain['key'] = 'value'
    self.assertIn(str({'key': 'value'}), stringifier(domain))