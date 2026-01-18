import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
class ConsoleUI:
    """
    A UI object that can ask true/false questions and post notifications on the
    console, to be used during key verification.
    """

    def __init__(self, opener):
        """
        @param opener: A no-argument callable which should open a console
            binary-mode file-like object to be used for reading and writing.
            This initializes the C{opener} attribute.
        @type opener: callable taking no arguments and returning a read/write
            file-like object
        """
        self.opener = opener

    def prompt(self, text):
        """
        Write the given text as a prompt to the console output, then read a
        result from the console input.

        @param text: Something to present to a user to solicit a yes or no
            response.
        @type text: L{bytes}

        @return: a L{Deferred} which fires with L{True} when the user answers
            'yes' and L{False} when the user answers 'no'.  It may errback if
            there were any I/O errors.
        """
        d = defer.succeed(None)

        def body(ignored):
            with closing(self.opener()) as f:
                f.write(text)
                while True:
                    answer = f.readline().strip().lower()
                    if answer == b'yes':
                        return True
                    elif answer == b'no':
                        return False
                    else:
                        f.write(b"Please type 'yes' or 'no': ")
        return d.addCallback(body)

    def warn(self, text):
        """
        Notify the user (non-interactively) of the provided text, by writing it
        to the console.

        @param text: Some information the user is to be made aware of.
        @type text: L{bytes}
        """
        try:
            with closing(self.opener()) as f:
                f.write(text)
        except Exception:
            log.failure('Failed to write to console')