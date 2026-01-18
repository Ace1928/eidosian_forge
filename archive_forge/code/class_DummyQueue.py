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
class DummyQueue:
    """
    A fake relay queue to use for testing.

    This queue doesn't keep track of which messages are waiting to be relayed
    or are in the process of being relayed.

    @ivar directory: See L{__init__}.
    """

    def __init__(self, directory):
        """
        @type directory: L{bytes}
        @param directory: The pathname of the directory holding messages in the
            queue.
        """
        self.directory = directory

    def done(self, message):
        """
        Remove a message from the queue.

        @type message: L{bytes}
        @param message: The base filename of a message.
        """
        message = os.path.basename(message)
        os.remove(self.getPath(message) + '-D')
        os.remove(self.getPath(message) + '-H')

    def getEnvelopeFile(self, message):
        """
        Get the envelope file for a message in the queue.

        @type message: L{bytes}
        @param message: The base filename of a message.

        @rtype: L{file}
        @return: The envelope file for the message.
        """
        return open(os.path.join(self.directory, message + '-H'), 'rb')

    def getPath(self, message):
        """
        Return the full base pathname of a message in the queue.

        @type message: L{bytes}
        @param message: The base filename of a message.

        @rtype: L{bytes}
        @return: The full base pathname of the message.
        """
        return os.path.join(self.directory, message)

    def createNewMessage(self):
        """
        Create a new message in the queue.

        @rtype: 2-L{tuple} of (E{1}) L{file}, (E{2}) L{FileMessage}
        @return: The envelope file and a message receiver for a new message in
            the queue.
        """
        fname = f'{time.time()}_{id(self)}'
        headerFile = open(os.path.join(self.directory, fname + '-H'), 'wb')
        tempFilename = os.path.join(self.directory, fname + '-C')
        finalFilename = os.path.join(self.directory, fname + '-D')
        messageFile = open(tempFilename, 'wb')
        return (headerFile, mail.mail.FileMessage(messageFile, tempFilename, finalFilename))

    def setWaiting(self, message):
        """
        Ignore the request to mark a message as waiting to be relayed.

        @type message: L{bytes}
        @param message: The base filename of a message.
        """
        pass