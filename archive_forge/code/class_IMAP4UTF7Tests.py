from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IMAP4UTF7Tests(TestCase):
    tests = [['Hello world', b'Hello world'], ['Hello & world', b'Hello &- world'], ['Helloÿworld', b'Hello&AP8-world'], ['ÿþýü', b'&AP8A,gD9APw-'], ['~peter/mail/日本語/台北', b'~peter/mail/&ZeVnLIqe-/&U,BTFw-']]

    def test_encodeWithErrors(self):
        """
        Specifying an error policy to C{unicode.encode} with the
        I{imap4-utf-7} codec should produce the same result as not
        specifying the error policy.
        """
        text = 'Hello world'
        self.assertEqual(text.encode('imap4-utf-7', 'strict'), text.encode('imap4-utf-7'))

    def test_decodeWithErrors(self):
        """
        Similar to L{test_encodeWithErrors}, but for C{bytes.decode}.
        """
        bytes = b'Hello world'
        self.assertEqual(bytes.decode('imap4-utf-7', 'strict'), bytes.decode('imap4-utf-7'))

    def test_encodeAmpersand(self):
        """
        Unicode strings that contain an ampersand (C{&}) can be
        encoded to bytes with the I{imap4-utf-7} codec.
        """
        text = '&Hello&½&'
        self.assertEqual(text.encode('imap4-utf-7'), b'&-Hello&-&AL0-&-')

    def test_decodeWithoutFinalASCIIShift(self):
        """
        An I{imap4-utf-7} encoded string that does not shift back to
        ASCII (i.e., it lacks a final C{-}) can be decoded.
        """
        self.assertEqual(b'&AL0'.decode('imap4-utf-7'), '½')

    def test_getreader(self):
        """
        C{codecs.getreader('imap4-utf-7')} returns the I{imap4-utf-7} stream
        reader class.
        """
        reader = codecs.getreader('imap4-utf-7')(BytesIO(b'Hello&AP8-world'))
        self.assertEqual(reader.read(), 'Helloÿworld')

    def test_getwriter(self):
        """
        C{codecs.getwriter('imap4-utf-7')} returns the I{imap4-utf-7} stream
        writer class.
        """
        output = BytesIO()
        writer = codecs.getwriter('imap4-utf-7')(output)
        writer.write('Helloÿworld')
        self.assertEqual(output.getvalue(), b'Hello&AP8-world')

    def test_encode(self):
        """
        The I{imap4-utf-7} can be used to encode a unicode string into a byte
        string according to the IMAP4 modified UTF-7 encoding rules.
        """
        for input, output in self.tests:
            self.assertEqual(input.encode('imap4-utf-7'), output)

    def test_decode(self):
        """
        The I{imap4-utf-7} can be used to decode a byte string into a unicode
        string according to the IMAP4 modified UTF-7 encoding rules.
        """
        for input, output in self.tests:
            self.assertEqual(input, output.decode('imap4-utf-7'))

    def test_printableSingletons(self):
        """
        The IMAP4 modified UTF-7 implementation encodes all printable
        characters which are in ASCII using the corresponding ASCII byte.
        """
        for o in chain(range(32, 38), range(39, 127)):
            charbyte = chr(o).encode()
            self.assertEqual(charbyte, chr(o).encode('imap4-utf-7'))
            self.assertEqual(chr(o), charbyte.decode('imap4-utf-7'))
        self.assertEqual('&'.encode('imap4-utf-7'), b'&-')
        self.assertEqual(b'&-'.decode('imap4-utf-7'), '&')