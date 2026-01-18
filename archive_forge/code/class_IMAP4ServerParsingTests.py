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
class IMAP4ServerParsingTests(SynchronousTestCase):
    """
    Test L{imap4.IMAP4Server}'s command parsing.
    """

    def setUp(self):
        self.transport = StringTransport()
        self.server = imap4.IMAP4Server()
        self.server.makeConnection(self.transport)
        self.transport.clear()

    def tearDown(self):
        self.server.connectionLost(failure.Failure(error.ConnectionDone()))

    def test_parseMethodExceptionLogged(self):
        """
        L{imap4.IMAP4Server} logs exceptions raised by parse methods.
        """

        class UnhandledException(Exception):
            """
            An unhandled exception.
            """

        def raisesValueError(line):
            raise UnhandledException
        self.server.parseState = 'command'
        self.server.parse_command = raisesValueError
        self.server.lineReceived(b'invalid')
        self.assertTrue(self.flushLoggedErrors(UnhandledException))

    def test_missingCommand(self):
        """
        L{imap4.IMAP4Server.parse_command} sends a C{BAD} response to
        a line that includes a tag but no command.
        """
        self.server.parse_command(b'001')
        self.assertEqual(self.transport.value(), b'001 BAD Missing command\r\n')
        self.server.connectionLost(failure.Failure(error.ConnectionDone('Done')))

    def test_emptyLine(self):
        """
        L{imap4.IMAP4Server.parse_command} sends a C{BAD} response to
        an empty line.
        """
        self.server.parse_command(b'')
        self.assertEqual(self.transport.value(), b'* BAD Null command\r\n')

    def assertParseExceptionResponse(self, exception, tag, expectedResponse):
        """
        Assert that the given exception results in the expected
        response.

        @param exception: The exception to raise.
        @type exception: L{Exception}

        @param tag: The IMAP tag.

        @type: L{bytes}

        @param expectedResponse: The expected bad response.
        @type expectedResponse: L{bytes}
        """

        def raises(tag, cmd, rest):
            raise exception
        self.server.dispatchCommand = raises
        self.server.parse_command(b' '.join([tag, b'invalid']))
        self.assertEqual(self.transport.value(), b' '.join([tag, expectedResponse]))

    def test_parsingRaisesIllegalClientResponse(self):
        """
        When a parsing method raises L{IllegalClientResponse}, the
        server sends a C{BAD} response.
        """
        self.assertParseExceptionResponse(imap4.IllegalClientResponse('client response'), b'001', b'BAD Illegal syntax: client response\r\n')

    def test_parsingRaisesIllegalOperationResponse(self):
        """
        When a parsing method raises L{IllegalOperation}, the server
        sends a C{NO} response.
        """
        self.assertParseExceptionResponse(imap4.IllegalOperation('operation'), b'001', b'NO Illegal operation: operation\r\n')

    def test_parsingRaisesIllegalMailboxEncoding(self):
        """
        When a parsing method raises L{IllegalMailboxEncoding}, the
        server sends a C{NO} response.
        """
        self.assertParseExceptionResponse(imap4.IllegalMailboxEncoding('encoding'), b'001', b'NO Illegal mailbox name: encoding\r\n')

    def test_unsupportedCommand(self):
        """
        L{imap4.IMAP4Server} responds to an unsupported command with a
        C{BAD} response.
        """
        self.server.lineReceived(b'001 HULLABALOO')
        self.assertEqual(self.transport.value(), b'001 BAD Unsupported command\r\n')

    def test_tooManyArgumentsForCommand(self):
        """
        L{imap4.IMAP4Server} responds with a C{BAD} response to a
        command with more arguments than expected.
        """
        self.server.lineReceived(b'001 LOGIN A B C')
        self.assertEqual(self.transport.value(), b'001 BAD Illegal syntax:' + b' Too many arguments for command: ' + repr(b'C').encode('utf-8') + b'\r\n')

    def assertCommandExceptionResponse(self, exception, tag, expectedResponse):
        """
        Assert that the given exception results in the expected
        response.

        @param exception: The exception to raise.
        @type exception: L{Exception}

        @param: The IMAP tag.

        @type: L{bytes}

        @param expectedResponse: The expected bad response.
        @type expectedResponse: L{bytes}
        """

        def raises(serverInstance, tag, user, passwd):
            raise exception
        self.assertEqual(self.server.state, 'unauth')
        self.server.unauth_LOGIN = (raises,) + self.server.unauth_LOGIN[1:]
        self.server.dispatchCommand(tag, b'LOGIN', b'user passwd')
        self.assertEqual(self.transport.value(), b' '.join([tag, expectedResponse]))

    def test_commandRaisesIllegalClientResponse(self):
        """
        When a command raises L{IllegalClientResponse}, the
        server sends a C{BAD} response.
        """
        self.assertCommandExceptionResponse(imap4.IllegalClientResponse('client response'), b'001', b'BAD Illegal syntax: client response\r\n')

    def test_commandRaisesIllegalOperationResponse(self):
        """
        When a command raises L{IllegalOperation}, the server sends a
        C{NO} response.
        """
        self.assertCommandExceptionResponse(imap4.IllegalOperation('operation'), b'001', b'NO Illegal operation: operation\r\n')

    def test_commandRaisesIllegalMailboxEncoding(self):
        """
        When a command raises L{IllegalMailboxEncoding}, the server
        sends a C{NO} response.
        """
        self.assertCommandExceptionResponse(imap4.IllegalMailboxEncoding('encoding'), b'001', b'NO Illegal mailbox name: encoding\r\n')

    def test_commandRaisesUnhandledException(self):
        """
        Wehn a command raises an unhandled exception, the server sends
        a C{BAD} response and logs the exception.
        """

        class UnhandledException(Exception):
            """
            An unhandled exception.
            """
        self.assertCommandExceptionResponse(UnhandledException('unhandled'), b'001', b'BAD Server error: unhandled\r\n')
        self.assertTrue(self.flushLoggedErrors(UnhandledException))

    def test_stringLiteralTooLong(self):
        """
        A string literal whose length exceeds the maximum allowed
        length results in a C{BAD} response.
        """
        self.server._literalStringLimit = 4
        self.server.lineReceived(b'001 LOGIN {5}\r\n')
        self.assertEqual(self.transport.value(), b'001 BAD Illegal syntax: Literal too long! I accept at most 4 octets\r\n')

    def test_arg_astringEmptyLine(self):
        """
        An empty string argument raises L{imap4.IllegalClientResponse}.
        """
        for empty in [b'', b'\r\n', b' ']:
            self.assertRaises(imap4.IllegalClientResponse, self.server.arg_astring, empty)

    def test_arg_astringUnmatchedQuotes(self):
        """
        An unmatched quote in a string argument raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_astring, b'"open')

    def test_arg_astringUnmatchedLiteralBraces(self):
        """
        An unmatched brace in a string literal's size raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_astring, b'{0')

    def test_arg_astringInvalidLiteralSize(self):
        """
        A non-integral string literal size raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_astring, b'{[object Object]}')

    def test_arg_atomEmptyLine(self):
        """
        An empty atom raises L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_atom, b'')

    def test_arg_atomMalformedAtom(self):
        """
        A malformed atom raises L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_atom, b' not an atom ')

    def test_arg_plistEmptyLine(self):
        """
        An empty parenthesized list raises L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_plist, b'')

    def test_arg_plistUnmatchedParentheses(self):
        """
        A parenthesized with unmatched parentheses raises
        L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_plist, b'(foo')
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_plist, b'foo)')

    def test_arg_literalEmptyLine(self):
        """
        An empty file literal raises L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_literal, b'')

    def test_arg_literalUnmatchedBraces(self):
        """
        A literal with unmatched braces raises
        L{IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_literal, b'{10')
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_literal, b'10}')

    def test_arg_literalInvalidLiteralSize(self):
        """
        A non-integral literal size raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_literal, b'{[object Object]}')

    def test_arg_seqsetReturnsRest(self):
        """
        A sequence set returns the unparsed portion of a line.
        """
        sequence = b'1:* blah blah blah'
        _, rest = self.server.arg_seqset(sequence)
        self.assertEqual(rest, b'blah blah blah')

    def test_arg_seqsetInvalidSequence(self):
        """
        An invalid sequence raises L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_seqset, b'x:y')

    def test_arg_flaglistOneFlag(self):
        """
        A single flag that is not contained in a list is parsed.
        """
        flag = b'flag'
        parsed, rest = self.server.arg_flaglist(flag)
        self.assertEqual(parsed, [flag])
        self.assertFalse(rest)

    def test_arg_flaglistMismatchedParentehses(self):
        """
        A list of flags with unmatched parentheses raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_flaglist, b'(invalid')

    def test_arg_flaglistMalformedFlag(self):
        """
        A list of flags that contains a malformed flag raises
        L{imap4.IllegalClientResponse}.
        """
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_flaglist, b'(first \x00)')
        self.assertRaises(imap4.IllegalClientResponse, self.server.arg_flaglist, b'(first \x00second)')

    def test_opt_plistMissingOpenParenthesis(self):
        """
        A line that does not begin with an open parenthesis (C{(}) is
        parsed as L{None}, and the remainder is the whole line.
        """
        line = b'not ('
        plist, remainder = self.server.opt_plist(line)
        self.assertIsNone(plist)
        self.assertEqual(remainder, line)

    def test_opt_datetimeMissingOpenQuote(self):
        """
        A line that does not begin with a double quote (C{"}) is
        parsed as L{None}, and the remainder is the whole line.
        """
        line = b'not "'
        dt, remainder = self.server.opt_datetime(line)
        self.assertIsNone(dt)
        self.assertEqual(remainder, line)

    def test_opt_datetimeMissingCloseQuote(self):
        """
        A line that does not have a closing double quote (C{"}) raises
        L{imap4.IllegalClientResponse}.
        """
        line = b'"21-Jul-2017 19:37:07 -0700'
        self.assertRaises(imap4.IllegalClientResponse, self.server.opt_datetime, line)

    def test_opt_charsetMissingIdentifier(self):
        """
        A line that contains C{CHARSET} but no character set
        identifier raises L{imap4.IllegalClientResponse}.
        """
        line = b'CHARSET'
        self.assertRaises(imap4.IllegalClientResponse, self.server.opt_charset, line)

    def test_opt_charsetEndOfLine(self):
        """
        A line that ends with a C{CHARSET} identifier is parsed as
        that identifier, and the remainder is the empty string.
        """
        line = b'CHARSET UTF-8'
        identifier, remainder = self.server.opt_charset(line)
        self.assertEqual(identifier, b'UTF-8')
        self.assertEqual(remainder, b'')

    def test_opt_charsetWithRemainder(self):
        """
        A line that has additional data after a C{CHARSET} identifier
        is parsed as that identifier, and the remainder is that
        additional data.
        """
        line = b'CHARSET UTF-8 remainder'
        identifier, remainder = self.server.opt_charset(line)
        self.assertEqual(identifier, b'UTF-8')
        self.assertEqual(remainder, b'remainder')