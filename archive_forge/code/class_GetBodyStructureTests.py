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
class GetBodyStructureTests(TestCase):
    """
    Tests for L{imap4.getBodyStructure}, a helper for constructing a list which
    directly corresponds to the wire information needed for a I{BODY} or
    I{BODYSTRUCTURE} response.
    """

    def test_singlePart(self):
        """
        L{imap4.getBodyStructure} accepts a L{IMessagePart} provider and returns
        a list giving the basic fields for the I{BODY} response for that
        message.
        """
        body = b'hello, world'
        major = 'image'
        minor = 'jpeg'
        charset = 'us-ascii'
        identifier = 'some kind of id'
        description = 'great justice'
        encoding = 'maximum'
        msg = FakeyMessage({'content-type': major + '/' + minor + '; charset=' + charset + '; x=y', 'content-id': identifier, 'content-description': description, 'content-transfer-encoding': encoding}, (), b'', body, 123, None)
        structure = imap4.getBodyStructure(msg)
        self.assertEqual([major, minor, ['charset', charset, 'x', 'y'], identifier, description, encoding, len(body)], structure)

    def test_emptyContentType(self):
        """
        L{imap4.getBodyStructure} returns L{None} for the major and
        minor MIME types of a L{IMessagePart} provider whose headers
        lack a C{Content-Type}, or have an empty value for it.
        """
        missing = FakeyMessage({}, (), b'', b'', 123, None)
        missingContentTypeStructure = imap4.getBodyStructure(missing)
        missingMajor, missingMinor = missingContentTypeStructure[:2]
        self.assertIs(None, missingMajor)
        self.assertIs(None, missingMinor)
        empty = FakeyMessage({'content-type': ''}, (), b'', b'', 123, None)
        emptyContentTypeStructure = imap4.getBodyStructure(empty)
        emptyMajor, emptyMinor = emptyContentTypeStructure[:2]
        self.assertIs(None, emptyMajor)
        self.assertIs(None, emptyMinor)
        newline = FakeyMessage({'content-type': '\n'}, (), b'', b'', 123, None)
        newlineContentTypeStructure = imap4.getBodyStructure(newline)
        newlineMajor, newlineMinor = newlineContentTypeStructure[:2]
        self.assertIs(None, newlineMajor)
        self.assertIs(None, newlineMinor)

    def test_onlyMajorContentType(self):
        """
        L{imap4.getBodyStructure} returns only a non-L{None} major
        MIME type for a L{IMessagePart} provider whose headers only
        have a main a C{Content-Type}.
        """
        main = FakeyMessage({'content-type': 'main'}, (), b'', b'', 123, None)
        mainStructure = imap4.getBodyStructure(main)
        mainMajor, mainMinor = mainStructure[:2]
        self.assertEqual(mainMajor, 'main')
        self.assertIs(mainMinor, None)

    def test_singlePartExtended(self):
        """
        L{imap4.getBodyStructure} returns a list giving the basic and extended
        fields for a I{BODYSTRUCTURE} response if passed C{True} for the
        C{extended} parameter.
        """
        body = b'hello, world'
        major = 'image'
        minor = 'jpeg'
        charset = 'us-ascii'
        identifier = 'some kind of id'
        description = 'great justice'
        encoding = 'maximum'
        md5 = 'abcdefabcdef'
        msg = FakeyMessage({'content-type': major + '/' + minor + '; charset=' + charset + '; x=y', 'content-id': identifier, 'content-description': description, 'content-transfer-encoding': encoding, 'content-md5': md5, 'content-disposition': 'attachment; name=foo; size=bar', 'content-language': 'fr', 'content-location': 'France'}, (), '', body, 123, None)
        structure = imap4.getBodyStructure(msg, extended=True)
        self.assertEqual([major, minor, ['charset', charset, 'x', 'y'], identifier, description, encoding, len(body), md5, ['attachment', ['name', 'foo', 'size', 'bar']], 'fr', 'France'], structure)

    def test_singlePartWithMissing(self):
        """
        For fields with no information contained in the message headers,
        L{imap4.getBodyStructure} fills in L{None} values in its result.
        """
        major = 'image'
        minor = 'jpeg'
        body = b'hello, world'
        msg = FakeyMessage({'content-type': major + '/' + minor}, (), b'', body, 123, None)
        structure = imap4.getBodyStructure(msg, extended=True)
        self.assertEqual([major, minor, None, None, None, None, len(body), None, None, None, None], structure)

    def test_textPart(self):
        """
        For a I{text/*} message, the number of lines in the message body are
        included after the common single-part basic fields.
        """
        body = b'hello, world\nhow are you?\ngoodbye\n'
        major = 'text'
        minor = 'jpeg'
        charset = 'us-ascii'
        identifier = 'some kind of id'
        description = 'great justice'
        encoding = 'maximum'
        msg = FakeyMessage({'content-type': major + '/' + minor + '; charset=' + charset + '; x=y', 'content-id': identifier, 'content-description': description, 'content-transfer-encoding': encoding}, (), b'', body, 123, None)
        structure = imap4.getBodyStructure(msg)
        self.assertEqual([major, minor, ['charset', charset, 'x', 'y'], identifier, description, encoding, len(body), len(body.splitlines())], structure)

    def test_rfc822Message(self):
        """
        For a I{message/rfc822} message, the common basic fields are followed
        by information about the contained message.
        """
        body = b'hello, world\nhow are you?\ngoodbye\n'
        major = 'text'
        minor = 'jpeg'
        charset = 'us-ascii'
        identifier = 'some kind of id'
        description = 'great justice'
        encoding = 'maximum'
        msg = FakeyMessage({'content-type': major + '/' + minor + '; charset=' + charset + '; x=y', 'from': 'Alice <alice@example.com>', 'to': 'Bob <bob@example.com>', 'content-id': identifier, 'content-description': description, 'content-transfer-encoding': encoding}, (), '', body, 123, None)
        container = FakeyMessage({'content-type': 'message/rfc822'}, (), b'', b'', 123, [msg])
        structure = imap4.getBodyStructure(container)
        self.assertEqual(['message', 'rfc822', None, None, None, None, 0, imap4.getEnvelope(msg), imap4.getBodyStructure(msg), 3], structure)

    def test_multiPart(self):
        """
        For a I{multipart/*} message, L{imap4.getBodyStructure} returns a list
        containing the body structure information for each part of the message
        followed by an element giving the MIME subtype of the message.
        """
        oneSubPart = FakeyMessage({'content-type': 'image/jpeg; x=y', 'content-id': 'some kind of id', 'content-description': 'great justice', 'content-transfer-encoding': 'maximum'}, (), b'', b'hello world', 123, None)
        anotherSubPart = FakeyMessage({'content-type': 'text/plain; charset=us-ascii'}, (), b'', b'some stuff', 321, None)
        container = FakeyMessage({'content-type': 'multipart/related'}, (), b'', b'', 555, [oneSubPart, anotherSubPart])
        self.assertEqual([imap4.getBodyStructure(oneSubPart), imap4.getBodyStructure(anotherSubPart), 'related'], imap4.getBodyStructure(container))

    def test_multiPartExtended(self):
        """
        When passed a I{multipart/*} message and C{True} for the C{extended}
        argument, L{imap4.getBodyStructure} includes extended structure
        information from the parts of the multipart message and extended
        structure information about the multipart message itself.
        """
        oneSubPart = FakeyMessage({b'content-type': b'image/jpeg; x=y', b'content-id': b'some kind of id', b'content-description': b'great justice', b'content-transfer-encoding': b'maximum'}, (), b'', b'hello world', 123, None)
        anotherSubPart = FakeyMessage({b'content-type': b'text/plain; charset=us-ascii'}, (), b'', b'some stuff', 321, None)
        container = FakeyMessage({'content-type': 'multipart/related; foo=bar', 'content-language': 'es', 'content-location': 'Spain', 'content-disposition': 'attachment; name=monkeys'}, (), b'', b'', 555, [oneSubPart, anotherSubPart])
        self.assertEqual([imap4.getBodyStructure(oneSubPart, extended=True), imap4.getBodyStructure(anotherSubPart, extended=True), 'related', ['foo', 'bar'], ['attachment', ['name', 'monkeys']], 'es', 'Spain'], imap4.getBodyStructure(container, extended=True))