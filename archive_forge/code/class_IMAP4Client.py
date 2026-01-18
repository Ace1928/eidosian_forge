import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
@implementer(IMailboxListener)
class IMAP4Client(basic.LineReceiver, policies.TimeoutMixin):
    """IMAP4 client protocol implementation

    @ivar state: A string representing the state the connection is currently
    in.
    """
    tags = None
    waiting = None
    queued = None
    tagID = 1
    state = None
    startedTLS = False
    timeout = 0
    _capCache = None
    _memoryFileLimit = 1024 * 1024 * 10
    authenticators = None
    STATUS_CODES = ('OK', 'NO', 'BAD', 'PREAUTH', 'BYE')
    STATUS_TRANSFORMATIONS = {'MESSAGES': int, 'RECENT': int, 'UNSEEN': int}
    context = None

    def __init__(self, contextFactory=None):
        self.tags = {}
        self.queued = []
        self.authenticators = {}
        self.context = contextFactory
        self._tag = None
        self._parts = None
        self._lastCmd = None

    def registerAuthenticator(self, auth):
        """
        Register a new form of authentication

        When invoking the authenticate() method of IMAP4Client, the first
        matching authentication scheme found will be used.  The ordering is
        that in which the server lists support authentication schemes.

        @type auth: Implementor of C{IClientAuthentication}
        @param auth: The object to use to perform the client
        side of this authentication scheme.
        """
        self.authenticators[auth.getName().upper()] = auth

    def rawDataReceived(self, data):
        if self.timeout > 0:
            self.resetTimeout()
        self._pendingSize -= len(data)
        if self._pendingSize > 0:
            self._pendingBuffer.write(data)
        else:
            passon = b''
            if self._pendingSize < 0:
                data, passon = (data[:self._pendingSize], data[self._pendingSize:])
            self._pendingBuffer.write(data)
            rest = self._pendingBuffer
            self._pendingBuffer = None
            self._pendingSize = None
            rest.seek(0, 0)
            self._parts.append(rest.read())
            self.setLineMode(passon.lstrip(b'\r\n'))

    def _setupForLiteral(self, rest, octets):
        self._pendingBuffer = self.messageFile(octets)
        self._pendingSize = octets
        if self._parts is None:
            self._parts = [rest, b'\r\n']
        else:
            self._parts.extend([rest, b'\r\n'])
        self.setRawMode()

    def connectionMade(self):
        if self.timeout > 0:
            self.setTimeout(self.timeout)

    def connectionLost(self, reason):
        """
        We are no longer connected
        """
        if self.timeout > 0:
            self.setTimeout(None)
        if self.queued is not None:
            queued = self.queued
            self.queued = None
            for cmd in queued:
                cmd.defer.errback(reason)
        if self.tags is not None:
            tags = self.tags
            self.tags = None
            for cmd in tags.values():
                if cmd is not None and cmd.defer is not None:
                    cmd.defer.errback(reason)

    def lineReceived(self, line):
        """
        Attempt to parse a single line from the server.

        @type line: L{bytes}
        @param line: The line from the server, without the line delimiter.

        @raise IllegalServerResponse: If the line or some part of the line
            does not represent an allowed message from the server at this time.
        """
        if self.timeout > 0:
            self.resetTimeout()
        lastPart = line.rfind(b'{')
        if lastPart != -1:
            lastPart = line[lastPart + 1:]
            if lastPart.endswith(b'}'):
                try:
                    octets = int(lastPart[:-1])
                except ValueError:
                    raise IllegalServerResponse(line)
                if self._parts is None:
                    self._tag, parts = line.split(None, 1)
                else:
                    parts = line
                self._setupForLiteral(parts, octets)
                return
        if self._parts is None:
            self._regularDispatch(line)
        else:
            self._parts.append(line)
            tag, rest = (self._tag, b''.join(self._parts))
            self._tag = self._parts = None
            self.dispatchCommand(tag, rest)

    def timeoutConnection(self):
        if self._lastCmd and self._lastCmd.defer is not None:
            d, self._lastCmd.defer = (self._lastCmd.defer, None)
            d.errback(TIMEOUT_ERROR)
        if self.queued:
            for cmd in self.queued:
                if cmd.defer is not None:
                    d, cmd.defer = (cmd.defer, d)
                    d.errback(TIMEOUT_ERROR)
        self.transport.loseConnection()

    def _regularDispatch(self, line):
        parts = line.split(None, 1)
        if len(parts) != 2:
            parts.append(b'')
        tag, rest = parts
        self.dispatchCommand(tag, rest)

    def messageFile(self, octets):
        """
        Create a file to which an incoming message may be written.

        @type octets: L{int}
        @param octets: The number of octets which will be written to the file

        @rtype: Any object which implements C{write(string)} and
        C{seek(int, int)}
        @return: A file-like object
        """
        if octets > self._memoryFileLimit:
            return tempfile.TemporaryFile()
        else:
            return BytesIO()

    def makeTag(self):
        tag = ('%0.4X' % self.tagID).encode('ascii')
        self.tagID += 1
        return tag

    def dispatchCommand(self, tag, rest):
        if self.state is None:
            f = self.response_UNAUTH
        else:
            f = getattr(self, 'response_' + self.state.upper(), None)
        if f:
            try:
                f(tag, rest)
            except BaseException:
                log.err()
                self.transport.loseConnection()
        else:
            log.err(f'Cannot dispatch: {self.state}, {tag!r}, {rest!r}')
            self.transport.loseConnection()

    def response_UNAUTH(self, tag, rest):
        if self.state is None:
            status, rest = rest.split(None, 1)
            if status.upper() == b'OK':
                self.state = 'unauth'
            elif status.upper() == b'PREAUTH':
                self.state = 'auth'
            else:
                self.transport.loseConnection()
                raise IllegalServerResponse(tag + b' ' + rest)
            b, e = (rest.find(b'['), rest.find(b']'))
            if b != -1 and e != -1:
                self.serverGreeting(self.__cbCapabilities(([parseNestedParens(rest[b + 1:e])], None)))
            else:
                self.serverGreeting(None)
        else:
            self._defaultHandler(tag, rest)

    def response_AUTH(self, tag, rest):
        self._defaultHandler(tag, rest)

    def _defaultHandler(self, tag, rest):
        if tag == b'*' or tag == b'+':
            if not self.waiting:
                self._extraInfo([parseNestedParens(rest)])
            else:
                cmd = self.tags[self.waiting]
                if tag == b'+':
                    cmd.continuation(rest)
                else:
                    cmd.lines.append(rest)
        else:
            try:
                cmd = self.tags[tag]
            except KeyError:
                self.transport.loseConnection()
                raise IllegalServerResponse(tag + b' ' + rest)
            else:
                status, line = rest.split(None, 1)
                if status == b'OK':
                    cmd.finish(rest, self._extraInfo)
                else:
                    cmd.defer.errback(IMAP4Exception(line))
                del self.tags[tag]
                self.waiting = None
                self._flushQueue()

    def _flushQueue(self):
        if self.queued:
            cmd = self.queued.pop(0)
            t = self.makeTag()
            self.tags[t] = cmd
            self.sendLine(cmd.format(t))
            self.waiting = t

    def _extraInfo(self, lines):
        flags = {}
        recent = exists = None
        for response in lines:
            elements = len(response)
            if elements == 1 and response[0] == [b'READ-ONLY']:
                self.modeChanged(False)
            elif elements == 1 and response[0] == [b'READ-WRITE']:
                self.modeChanged(True)
            elif elements == 2 and response[1] == b'EXISTS':
                exists = int(response[0])
            elif elements == 2 and response[1] == b'RECENT':
                recent = int(response[0])
            elif elements == 3 and response[1] == b'FETCH':
                mId = int(response[0])
                values, _ = self._parseFetchPairs(response[2])
                flags.setdefault(mId, []).extend(values.get('FLAGS', ()))
            else:
                log.msg(f'Unhandled unsolicited response: {response}')
        if flags:
            self.flagsChanged(flags)
        if recent is not None or exists is not None:
            self.newMessages(exists, recent)

    def sendCommand(self, cmd):
        cmd.defer = defer.Deferred()
        if self.waiting:
            self.queued.append(cmd)
            return cmd.defer
        t = self.makeTag()
        self.tags[t] = cmd
        self.sendLine(cmd.format(t))
        self.waiting = t
        self._lastCmd = cmd
        return cmd.defer

    def getCapabilities(self, useCache=1):
        """
        Request the capabilities available on this server.

        This command is allowed in any state of connection.

        @type useCache: C{bool}
        @param useCache: Specify whether to use the capability-cache or to
        re-retrieve the capabilities from the server.  Server capabilities
        should never change, so for normal use, this flag should never be
        false.

        @rtype: L{Deferred}
        @return: A deferred whose callback will be invoked with a
        dictionary mapping capability types to lists of supported
        mechanisms, or to None if a support list is not applicable.
        """
        if useCache and self._capCache is not None:
            return defer.succeed(self._capCache)
        cmd = b'CAPABILITY'
        resp = (b'CAPABILITY',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbCapabilities)
        return d

    def __cbCapabilities(self, result):
        lines, tagline = result
        caps = {}
        for rest in lines:
            for cap in rest[1:]:
                parts = cap.split(b'=', 1)
                if len(parts) == 1:
                    category, value = (parts[0], None)
                else:
                    category, value = parts
                caps.setdefault(category, []).append(value)
        for category in caps:
            if caps[category] == [None]:
                caps[category] = None
        self._capCache = caps
        return caps

    def logout(self):
        """
        Inform the server that we are done with the connection.

        This command is allowed in any state of connection.

        @rtype: L{Deferred}
        @return: A deferred whose callback will be invoked with None
        when the proper server acknowledgement has been received.
        """
        d = self.sendCommand(Command(b'LOGOUT', wantResponse=(b'BYE',)))
        d.addCallback(self.__cbLogout)
        return d

    def __cbLogout(self, result):
        lines, tagline = result
        self.transport.loseConnection()
        return None

    def noop(self):
        """
        Perform no operation.

        This command is allowed in any state of connection.

        @rtype: L{Deferred}
        @return: A deferred whose callback will be invoked with a list
        of untagged status updates the server responds with.
        """
        d = self.sendCommand(Command(b'NOOP'))
        d.addCallback(self.__cbNoop)
        return d

    def __cbNoop(self, result):
        lines, tagline = result
        return lines

    def startTLS(self, contextFactory=None):
        """
        Initiates a 'STARTTLS' request and negotiates the TLS / SSL
        Handshake.

        @param contextFactory: The TLS / SSL Context Factory to
        leverage.  If the contextFactory is None the IMAP4Client will
        either use the current TLS / SSL Context Factory or attempt to
        create a new one.

        @type contextFactory: C{ssl.ClientContextFactory}

        @return: A Deferred which fires when the transport has been
        secured according to the given contextFactory, or which fails
        if the transport cannot be secured.
        """
        assert not self.startedTLS, 'Client and Server are currently communicating via TLS'
        if contextFactory is None:
            contextFactory = self._getContextFactory()
        if contextFactory is None:
            return defer.fail(IMAP4Exception('IMAP4Client requires a TLS context to initiate the STARTTLS handshake'))
        if b'STARTTLS' not in self._capCache:
            return defer.fail(IMAP4Exception('Server does not support secure communication via TLS / SSL'))
        tls = interfaces.ITLSTransport(self.transport, None)
        if tls is None:
            return defer.fail(IMAP4Exception('IMAP4Client transport does not implement interfaces.ITLSTransport'))
        d = self.sendCommand(Command(b'STARTTLS'))
        d.addCallback(self._startedTLS, contextFactory)
        d.addCallback(lambda _: self.getCapabilities())
        return d

    def authenticate(self, secret):
        """
        Attempt to enter the authenticated state with the server

        This command is allowed in the Non-Authenticated state.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if the authentication
        succeeds and whose errback will be invoked otherwise.
        """
        if self._capCache is None:
            d = self.getCapabilities()
        else:
            d = defer.succeed(self._capCache)
        d.addCallback(self.__cbAuthenticate, secret)
        return d

    def __cbAuthenticate(self, caps, secret):
        auths = caps.get(b'AUTH', ())
        for scheme in auths:
            if scheme.upper() in self.authenticators:
                cmd = Command(b'AUTHENTICATE', scheme, (), self.__cbContinueAuth, scheme, secret)
                return self.sendCommand(cmd)
        if self.startedTLS:
            return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
        else:

            def ebStartTLS(err):
                err.trap(IMAP4Exception)
                return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
            d = self.startTLS()
            d.addErrback(ebStartTLS)
            d.addCallback(lambda _: self.getCapabilities())
            d.addCallback(self.__cbAuthTLS, secret)
            return d

    def __cbContinueAuth(self, rest, scheme, secret):
        try:
            chal = decodebytes(rest + b'\n')
        except binascii.Error:
            self.sendLine(b'*')
            raise IllegalServerResponse(rest)
        else:
            auth = self.authenticators[scheme]
            chal = auth.challengeResponse(secret, chal)
            self.sendLine(encodebytes(chal).strip())

    def __cbAuthTLS(self, caps, secret):
        auths = caps.get(b'AUTH', ())
        for scheme in auths:
            if scheme.upper() in self.authenticators:
                cmd = Command(b'AUTHENTICATE', scheme, (), self.__cbContinueAuth, scheme, secret)
                return self.sendCommand(cmd)
        raise NoSupportedAuthentication(auths, self.authenticators.keys())

    def login(self, username, password):
        """
        Authenticate with the server using a username and password

        This command is allowed in the Non-Authenticated state.  If the
        server supports the STARTTLS capability and our transport supports
        TLS, TLS is negotiated before the login command is issued.

        A more secure way to log in is to use C{startTLS} or
        C{authenticate} or both.

        @type username: L{str}
        @param username: The username to log in with

        @type password: L{str}
        @param password: The password to log in with

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if login is successful
        and whose errback is invoked otherwise.
        """
        d = maybeDeferred(self.getCapabilities)
        d.addCallback(self.__cbLoginCaps, username, password)
        return d

    def serverGreeting(self, caps):
        """
        Called when the server has sent us a greeting.

        @type caps: C{dict}
        @param caps: Capabilities the server advertised in its greeting.
        """

    def _getContextFactory(self):
        if self.context is not None:
            return self.context
        try:
            from twisted.internet import ssl
        except ImportError:
            return None
        else:
            return ssl.ClientContextFactory()

    def __cbLoginCaps(self, capabilities, username, password):
        tryTLS = b'STARTTLS' in capabilities
        tlsableTransport = interfaces.ITLSTransport(self.transport, None) is not None
        nontlsTransport = interfaces.ISSLTransport(self.transport, None) is None
        if not self.startedTLS and tryTLS and tlsableTransport and nontlsTransport:
            d = self.startTLS()
            d.addCallbacks(self.__cbLoginTLS, self.__ebLoginTLS, callbackArgs=(username, password))
            return d
        else:
            if nontlsTransport:
                log.msg('Server has no TLS support. logging in over cleartext!')
            args = b' '.join((_quote(username), _quote(password)))
            return self.sendCommand(Command(b'LOGIN', args))

    def _startedTLS(self, result, context):
        self.transport.startTLS(context)
        self._capCache = None
        self.startedTLS = True
        return result

    def __cbLoginTLS(self, result, username, password):
        args = b' '.join((_quote(username), _quote(password)))
        return self.sendCommand(Command(b'LOGIN', args))

    def __ebLoginTLS(self, failure):
        log.err(failure)
        return failure

    def namespace(self):
        """
        Retrieve information about the namespaces available to this account

        This command is allowed in the Authenticated and Selected states.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with namespace
        information.  An example of this information is::

            [[['', '/']], [], []]

        which indicates a single personal namespace called '' with '/'
        as its hierarchical delimiter, and no shared or user namespaces.
        """
        cmd = b'NAMESPACE'
        resp = (b'NAMESPACE',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbNamespace)
        return d

    def __cbNamespace(self, result):
        lines, last = result

        def _prepareNamespaceOrDelimiter(namespaceList):
            return [element.decode('imap4-utf-7') for element in namespaceList]
        for parts in lines:
            if len(parts) == 4 and parts[0] == b'NAMESPACE':
                return [[] if pairOrNone is None else [_prepareNamespaceOrDelimiter(value) for value in pairOrNone] for pairOrNone in parts[1:]]
        log.err('No NAMESPACE response to NAMESPACE command')
        return [[], [], []]

    def select(self, mailbox):
        """
        Select a mailbox

        This command is allowed in the Authenticated and Selected states.

        @type mailbox: L{str}
        @param mailbox: The name of the mailbox to select

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with mailbox
        information if the select is successful and whose errback is
        invoked otherwise.  Mailbox information consists of a dictionary
        with the following L{str} keys and values::

                FLAGS: A list of strings containing the flags settable on
                        messages in this mailbox.

                EXISTS: An integer indicating the number of messages in this
                        mailbox.

                RECENT: An integer indicating the number of "recent"
                        messages in this mailbox.

                UNSEEN: The message sequence number (an integer) of the
                        first unseen message in the mailbox.

                PERMANENTFLAGS: A list of strings containing the flags that
                        can be permanently set on messages in this mailbox.

                UIDVALIDITY: An integer uniquely identifying this mailbox.
        """
        cmd = b'SELECT'
        args = _prepareMailboxName(mailbox)
        resp = ('FLAGS', 'EXISTS', 'RECENT', 'UNSEEN', 'PERMANENTFLAGS', 'UIDVALIDITY')
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbSelect, 1)
        return d

    def examine(self, mailbox):
        """
        Select a mailbox in read-only mode

        This command is allowed in the Authenticated and Selected states.

        @type mailbox: L{str}
        @param mailbox: The name of the mailbox to examine

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with mailbox
        information if the examine is successful and whose errback
        is invoked otherwise.  Mailbox information consists of a dictionary
        with the following keys and values::

            'FLAGS': A list of strings containing the flags settable on
                        messages in this mailbox.

            'EXISTS': An integer indicating the number of messages in this
                        mailbox.

            'RECENT': An integer indicating the number of "recent"
                        messages in this mailbox.

            'UNSEEN': An integer indicating the number of messages not
                        flagged \\Seen in this mailbox.

            'PERMANENTFLAGS': A list of strings containing the flags that
                        can be permanently set on messages in this mailbox.

            'UIDVALIDITY': An integer uniquely identifying this mailbox.
        """
        cmd = b'EXAMINE'
        args = _prepareMailboxName(mailbox)
        resp = (b'FLAGS', b'EXISTS', b'RECENT', b'UNSEEN', b'PERMANENTFLAGS', b'UIDVALIDITY')
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbSelect, 0)
        return d

    def _intOrRaise(self, value, phrase):
        """
        Parse C{value} as an integer and return the result or raise
        L{IllegalServerResponse} with C{phrase} as an argument if C{value}
        cannot be parsed as an integer.
        """
        try:
            return int(value)
        except ValueError:
            raise IllegalServerResponse(phrase)

    def __cbSelect(self, result, rw):
        """
        Handle lines received in response to a SELECT or EXAMINE command.

        See RFC 3501, section 6.3.1.
        """
        lines, tagline = result
        datum = {'READ-WRITE': rw}
        lines.append(parseNestedParens(tagline))
        for split in lines:
            if len(split) > 0 and split[0].upper() == b'OK':
                content = split[1]
                if isinstance(content, list):
                    key = content[0]
                else:
                    key = content
                key = key.upper()
                if key == b'READ-ONLY':
                    datum['READ-WRITE'] = False
                elif key == b'READ-WRITE':
                    datum['READ-WRITE'] = True
                elif key == b'UIDVALIDITY':
                    datum['UIDVALIDITY'] = self._intOrRaise(content[1], split)
                elif key == b'UNSEEN':
                    datum['UNSEEN'] = self._intOrRaise(content[1], split)
                elif key == b'UIDNEXT':
                    datum['UIDNEXT'] = self._intOrRaise(content[1], split)
                elif key == b'PERMANENTFLAGS':
                    datum['PERMANENTFLAGS'] = tuple((nativeString(flag) for flag in content[1]))
                else:
                    log.err(f'Unhandled SELECT response (2): {split}')
            elif len(split) == 2:
                if split[0].upper() == b'FLAGS':
                    datum['FLAGS'] = tuple((nativeString(flag) for flag in split[1]))
                elif isinstance(split[1], bytes):
                    if split[1].upper() == b'EXISTS':
                        datum['EXISTS'] = self._intOrRaise(split[0], split)
                    elif split[1].upper() == b'RECENT':
                        datum['RECENT'] = self._intOrRaise(split[0], split)
                    else:
                        log.err(f'Unhandled SELECT response (0): {split}')
                else:
                    log.err(f'Unhandled SELECT response (1): {split}')
            else:
                log.err(f'Unhandled SELECT response (4): {split}')
        return datum

    def create(self, name):
        """
        Create a new mailbox on the server

        This command is allowed in the Authenticated and Selected states.

        @type name: L{str}
        @param name: The name of the mailbox to create.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if the mailbox creation
        is successful and whose errback is invoked otherwise.
        """
        return self.sendCommand(Command(b'CREATE', _prepareMailboxName(name)))

    def delete(self, name):
        """
        Delete a mailbox

        This command is allowed in the Authenticated and Selected states.

        @type name: L{str}
        @param name: The name of the mailbox to delete.

        @rtype: L{Deferred}
        @return: A deferred whose calblack is invoked if the mailbox is
        deleted successfully and whose errback is invoked otherwise.
        """
        return self.sendCommand(Command(b'DELETE', _prepareMailboxName(name)))

    def rename(self, oldname, newname):
        """
        Rename a mailbox

        This command is allowed in the Authenticated and Selected states.

        @type oldname: L{str}
        @param oldname: The current name of the mailbox to rename.

        @type newname: L{str}
        @param newname: The new name to give the mailbox.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if the rename is
        successful and whose errback is invoked otherwise.
        """
        oldname = _prepareMailboxName(oldname)
        newname = _prepareMailboxName(newname)
        return self.sendCommand(Command(b'RENAME', b' '.join((oldname, newname))))

    def subscribe(self, name):
        """
        Add a mailbox to the subscription list

        This command is allowed in the Authenticated and Selected states.

        @type name: L{str}
        @param name: The mailbox to mark as 'active' or 'subscribed'

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if the subscription
        is successful and whose errback is invoked otherwise.
        """
        return self.sendCommand(Command(b'SUBSCRIBE', _prepareMailboxName(name)))

    def unsubscribe(self, name):
        """
        Remove a mailbox from the subscription list

        This command is allowed in the Authenticated and Selected states.

        @type name: L{str}
        @param name: The mailbox to unsubscribe

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked if the unsubscription
        is successful and whose errback is invoked otherwise.
        """
        return self.sendCommand(Command(b'UNSUBSCRIBE', _prepareMailboxName(name)))

    def list(self, reference, wildcard):
        """
        List a subset of the available mailboxes

        This command is allowed in the Authenticated and Selected
        states.

        @type reference: L{str}
        @param reference: The context in which to interpret
            C{wildcard}

        @type wildcard: L{str}
        @param wildcard: The pattern of mailbox names to match,
            optionally including either or both of the '*' and '%'
            wildcards.  '*' will match zero or more characters and
            cross hierarchical boundaries.  '%' will also match zero
            or more characters, but is limited to a single
            hierarchical level.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of
            L{tuple}s, the first element of which is a L{tuple} of
            mailbox flags, the second element of which is the
            hierarchy delimiter for this mailbox, and the third of
            which is the mailbox name; if the command is unsuccessful,
            the deferred's errback is invoked instead.  B{NB}: the
            delimiter and the mailbox name are L{str}s.
        """
        cmd = b'LIST'
        args = f'"{reference}" "{wildcard}"'.encode('imap4-utf-7')
        resp = (b'LIST',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbList, b'LIST')
        return d

    def lsub(self, reference, wildcard):
        """
        List a subset of the subscribed available mailboxes

        This command is allowed in the Authenticated and Selected states.

        The parameters and returned object are the same as for the L{list}
        method, with one slight difference: Only mailboxes which have been
        subscribed can be included in the resulting list.
        """
        cmd = b'LSUB'
        encodedReference = reference.encode('ascii')
        encodedWildcard = wildcard.encode('imap4-utf-7')
        args = b''.join([b'"', encodedReference, b'" "', encodedWildcard, b'"'])
        resp = (b'LSUB',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbList, b'LSUB')
        return d

    def __cbList(self, result, command):
        lines, last = result
        results = []
        for parts in lines:
            if len(parts) == 4 and parts[0] == command:
                parts[1] = tuple((nativeString(flag) for flag in parts[1]))
                parts[2] = parts[2].decode('imap4-utf-7')
                parts[3] = parts[3].decode('imap4-utf-7')
                results.append(tuple(parts[1:]))
        return results
    _statusNames = {name: name.encode('ascii') for name in ('MESSAGES', 'RECENT', 'UIDNEXT', 'UIDVALIDITY', 'UNSEEN')}

    def status(self, mailbox, *names):
        """
        Retrieve the status of the given mailbox

        This command is allowed in the Authenticated and Selected states.

        @type mailbox: L{str}
        @param mailbox: The name of the mailbox to query

        @type names: L{bytes}
        @param names: The status names to query.  These may be any number of:
            C{'MESSAGES'}, C{'RECENT'}, C{'UIDNEXT'}, C{'UIDVALIDITY'}, and
            C{'UNSEEN'}.

        @rtype: L{Deferred}
        @return: A deferred which fires with the status information if the
            command is successful and whose errback is invoked otherwise.  The
            status information is in the form of a C{dict}.  Each element of
            C{names} is a key in the dictionary.  The value for each key is the
            corresponding response from the server.
        """
        cmd = b'STATUS'
        preparedMailbox = _prepareMailboxName(mailbox)
        try:
            names = b' '.join((self._statusNames[name] for name in names))
        except KeyError:
            raise ValueError(f'Unknown names: {set(names) - set(self._statusNames)!r}')
        args = b''.join([preparedMailbox, b' (', names, b')'])
        resp = (b'STATUS',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbStatus)
        return d

    def __cbStatus(self, result):
        lines, last = result
        status = {}
        for parts in lines:
            if parts[0] == b'STATUS':
                items = parts[2]
                items = [items[i:i + 2] for i in range(0, len(items), 2)]
                for k, v in items:
                    try:
                        status[nativeString(k)] = v
                    except UnicodeDecodeError:
                        raise IllegalServerResponse(repr(items))
        for k in status.keys():
            t = self.STATUS_TRANSFORMATIONS.get(k)
            if t:
                try:
                    status[k] = t(status[k])
                except Exception as e:
                    raise IllegalServerResponse('(' + k + ' ' + status[k] + '): ' + str(e))
        return status

    def append(self, mailbox, message, flags=(), date=None):
        """
        Add the given message to the given mailbox.

        This command is allowed in the Authenticated and Selected states.

        @type mailbox: L{str}
        @param mailbox: The mailbox to which to add this message.

        @type message: Any file-like object opened in B{binary mode}.
        @param message: The message to add, in RFC822 format.  Newlines
        in this file should be \\r\\n-style.

        @type flags: Any iterable of L{str}
        @param flags: The flags to associated with this message.

        @type date: L{str}
        @param date: The date to associate with this message.  This should
        be of the format DD-MM-YYYY HH:MM:SS +/-HHMM.  For example, in
        Eastern Standard Time, on July 1st 2004 at half past 1 PM,
        "01-07-2004 13:30:00 -0500".

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked when this command
        succeeds or whose errback is invoked if it fails.
        """
        message.seek(0, 2)
        L = message.tell()
        message.seek(0, 0)
        if date:
            date = networkString(' "%s"' % nativeString(date))
        else:
            date = b''
        encodedFlags = [networkString(flag) for flag in flags]
        cmd = b'%b (%b)%b {%d}' % (_prepareMailboxName(mailbox), b' '.join(encodedFlags), date, L)
        d = self.sendCommand(Command(b'APPEND', cmd, (), self.__cbContinueAppend, message))
        return d

    def __cbContinueAppend(self, lines, message):
        s = basic.FileSender()
        return s.beginFileTransfer(message, self.transport, None).addCallback(self.__cbFinishAppend)

    def __cbFinishAppend(self, foo):
        self.sendLine(b'')

    def check(self):
        """
        Tell the server to perform a checkpoint

        This command is allowed in the Selected state.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked when this command
        succeeds or whose errback is invoked if it fails.
        """
        return self.sendCommand(Command(b'CHECK'))

    def close(self):
        """
        Return the connection to the Authenticated state.

        This command is allowed in the Selected state.

        Issuing this command will also remove all messages flagged \\Deleted
        from the selected mailbox if it is opened in read-write mode,
        otherwise it indicates success by no messages are removed.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked when the command
        completes successfully or whose errback is invoked if it fails.
        """
        return self.sendCommand(Command(b'CLOSE'))

    def expunge(self):
        """
        Return the connection to the Authenticate state.

        This command is allowed in the Selected state.

        Issuing this command will perform the same actions as issuing the
        close command, but will also generate an 'expunge' response for
        every message deleted.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of the
        'expunge' responses when this command is successful or whose errback
        is invoked otherwise.
        """
        cmd = b'EXPUNGE'
        resp = (b'EXPUNGE',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbExpunge)
        return d

    def __cbExpunge(self, result):
        lines, last = result
        ids = []
        for parts in lines:
            if len(parts) == 2 and parts[1] == b'EXPUNGE':
                ids.append(self._intOrRaise(parts[0], parts))
        return ids

    def search(self, *queries, uid=False):
        """
        Search messages in the currently selected mailbox

        This command is allowed in the Selected state.

        Any non-zero number of queries are accepted by this method, as returned
        by the C{Query}, C{Or}, and C{Not} functions.

        @param uid: if true, the server is asked to return message UIDs instead
            of message sequence numbers.
        @type uid: L{bool}

        @rtype: L{Deferred}
        @return: A deferred whose callback will be invoked with a list of all
            the message sequence numbers return by the search, or whose errback
            will be invoked if there is an error.
        """
        queries = [query.encode('charmap') for query in queries]
        cmd = b'UID SEARCH' if uid else b'SEARCH'
        args = b' '.join(queries)
        d = self.sendCommand(Command(cmd, args, wantResponse=(cmd,)))
        d.addCallback(self.__cbSearch)
        return d

    def __cbSearch(self, result):
        lines, end = result
        ids = []
        for parts in lines:
            if len(parts) > 0 and parts[0] == b'SEARCH':
                ids.extend([self._intOrRaise(p, parts) for p in parts[1:]])
        return ids

    def fetchUID(self, messages, uid=0):
        """
        Retrieve the unique identifier for one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message sequence numbers to unique message identifiers, or whose
        errback is invoked if there is an error.
        """
        return self._fetch(messages, useUID=uid, uid=1)

    def fetchFlags(self, messages, uid=0):
        """
        Retrieve the flags for one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve flags.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to lists of flags, or whose errback is invoked if
        there is an error.
        """
        return self._fetch(messages, useUID=uid, flags=1)

    def fetchInternalDate(self, messages, uid=0):
        """
        Retrieve the internal date associated with one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve the internal date.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to date strings, or whose errback is invoked
        if there is an error.  Date strings take the format of
        "day-month-year time timezone".
        """
        return self._fetch(messages, useUID=uid, internaldate=1)

    def fetchEnvelope(self, messages, uid=0):
        """
        Retrieve the envelope data for one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve envelope
            data.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of
            message numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict
            mapping message numbers to envelope data, or whose errback
            is invoked if there is an error.  Envelope data consists
            of a sequence of the date, subject, from, sender,
            reply-to, to, cc, bcc, in-reply-to, and message-id header
            fields.  The date, subject, in-reply-to, and message-id
            fields are L{str}, while the from, sender, reply-to, to,
            cc, and bcc fields contain address data as L{str}s.
            Address data consists of a sequence of name, source route,
            mailbox name, and hostname.  Fields which are not present
            for a particular address may be L{None}.
        """
        return self._fetch(messages, useUID=uid, envelope=1)

    def fetchBodyStructure(self, messages, uid=0):
        """
        Retrieve the structure of the body of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve body structure
        data.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to body structure data, or whose errback is invoked
        if there is an error.  Body structure data describes the MIME-IMB
        format of a message and consists of a sequence of mime type, mime
        subtype, parameters, content id, description, encoding, and size.
        The fields following the size field are variable: if the mime
        type/subtype is message/rfc822, the contained message's envelope
        information, body structure data, and number of lines of text; if
        the mime type is text, the number of lines of text.  Extension fields
        may also be included; if present, they are: the MD5 hash of the body,
        body disposition, body language.
        """
        return self._fetch(messages, useUID=uid, bodystructure=1)

    def fetchSimplifiedBody(self, messages, uid=0):
        """
        Retrieve the simplified body structure of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: C{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to body data, or whose errback is invoked
        if there is an error.  The simplified body structure is the same
        as the body structure, except that extension fields will never be
        present.
        """
        return self._fetch(messages, useUID=uid, body=1)

    def fetchMessage(self, messages, uid=0):
        """
        Retrieve one or more entire messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: C{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}

        @return: A L{Deferred} which will fire with a C{dict} mapping message
            sequence numbers to C{dict}s giving message data for the
            corresponding message.  If C{uid} is true, the inner dictionaries
            have a C{'UID'} key mapped to a L{str} giving the UID for the
            message.  The text of the message is a L{str} associated with the
            C{'RFC822'} key in each dictionary.
        """
        return self._fetch(messages, useUID=uid, rfc822=1)

    def fetchHeaders(self, messages, uid=0):
        """
        Retrieve headers of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to dicts of message headers, or whose errback is
        invoked if there is an error.
        """
        return self._fetch(messages, useUID=uid, rfc822header=1)

    def fetchBody(self, messages, uid=0):
        """
        Retrieve body text of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to file-like objects containing body text, or whose
        errback is invoked if there is an error.
        """
        return self._fetch(messages, useUID=uid, rfc822text=1)

    def fetchSize(self, messages, uid=0):
        """
        Retrieve the size, in octets, of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to sizes, or whose errback is invoked if there is
        an error.
        """
        return self._fetch(messages, useUID=uid, rfc822size=1)

    def fetchFull(self, messages, uid=0):
        """
        Retrieve several different fields of one or more messages

        This command is allowed in the Selected state.  This is equivalent
        to issuing all of the C{fetchFlags}, C{fetchInternalDate},
        C{fetchSize}, C{fetchEnvelope}, and C{fetchSimplifiedBody}
        functions.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to dict of the retrieved data values, or whose
        errback is invoked if there is an error.  They dictionary keys
        are "flags", "date", "size", "envelope", and "body".
        """
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1, envelope=1, body=1)

    def fetchAll(self, messages, uid=0):
        """
        Retrieve several different fields of one or more messages

        This command is allowed in the Selected state.  This is equivalent
        to issuing all of the C{fetchFlags}, C{fetchInternalDate},
        C{fetchSize}, and C{fetchEnvelope} functions.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to dict of the retrieved data values, or whose
        errback is invoked if there is an error.  They dictionary keys
        are "flags", "date", "size", and "envelope".
        """
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1, envelope=1)

    def fetchFast(self, messages, uid=0):
        """
        Retrieve several different fields of one or more messages

        This command is allowed in the Selected state.  This is equivalent
        to issuing all of the C{fetchFlags}, C{fetchInternalDate}, and
        C{fetchSize} functions.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to dict of the retrieved data values, or whose
        errback is invoked if there is an error.  They dictionary keys are
        "flags", "date", and "size".
        """
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1)

    def _parseFetchPairs(self, fetchResponseList):
        """
        Given the result of parsing a single I{FETCH} response, construct a
        L{dict} mapping response keys to response values.

        @param fetchResponseList: The result of parsing a I{FETCH} response
            with L{parseNestedParens} and extracting just the response data
            (that is, just the part that comes after C{"FETCH"}).  The form
            of this input (and therefore the output of this method) is very
            disagreeable.  A valuable improvement would be to enumerate the
            possible keys (representing them as structured objects of some
            sort) rather than using strings and tuples of tuples of strings
            and so forth.  This would allow the keys to be documented more
            easily and would allow for a much simpler application-facing API
            (one not based on looking up somewhat hard to predict keys in a
            dict).  Since C{fetchResponseList} notionally represents a
            flattened sequence of pairs (identifying keys followed by their
            associated values), collapsing such complex elements of this
            list as C{["BODY", ["HEADER.FIELDS", ["SUBJECT"]]]} into a
            single object would also greatly simplify the implementation of
            this method.

        @return: A C{dict} of the response data represented by C{pairs}.  Keys
            in this dictionary are things like C{"RFC822.TEXT"}, C{"FLAGS"}, or
            C{("BODY", ("HEADER.FIELDS", ("SUBJECT",)))}.  Values are entirely
            dependent on the key with which they are associated, but retain the
            same structured as produced by L{parseNestedParens}.
        """

        def nativeStringResponse(thing):
            if isinstance(thing, bytes):
                return thing.decode('charmap')
            elif isinstance(thing, list):
                return [nativeStringResponse(subthing) for subthing in thing]
        values = {}
        unstructured = []
        responseParts = iter(fetchResponseList)
        while True:
            try:
                key = next(responseParts)
            except StopIteration:
                break
            try:
                value = next(responseParts)
            except StopIteration:
                raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
            if key not in (b'BODY', b'BODY.PEEK'):
                hasSection = False
            elif not isinstance(value, list):
                hasSection = False
            elif len(value) > 2:
                hasSection = False
            elif value and isinstance(value[0], list):
                hasSection = False
            else:
                hasSection = True
            key = nativeString(key)
            unstructured.append(key)
            if hasSection:
                if len(value) < 2:
                    value = [nativeString(v) for v in value]
                    unstructured.append(value)
                    key = (key, tuple(value))
                else:
                    valueHead = nativeString(value[0])
                    valueTail = [nativeString(v) for v in value[1]]
                    unstructured.append([valueHead, valueTail])
                    key = (key, (valueHead, tuple(valueTail)))
                try:
                    value = next(responseParts)
                except StopIteration:
                    raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
                if value.startswith(b'<') and value.endswith(b'>'):
                    try:
                        int(value[1:-1])
                    except ValueError:
                        pass
                    else:
                        value = nativeString(value)
                        unstructured.append(value)
                        key = key + (value,)
                        try:
                            value = next(responseParts)
                        except StopIteration:
                            raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
            value = nativeStringResponse(value)
            unstructured.append(value)
            values[key] = value
        return (values, unstructured)

    def _cbFetch(self, result, requestedParts, structured):
        lines, last = result
        info = {}
        for parts in lines:
            if len(parts) == 3 and parts[1] == b'FETCH':
                id = self._intOrRaise(parts[0], parts)
                if id not in info:
                    info[id] = [parts[2]]
                else:
                    info[id][0].extend(parts[2])
        results = {}
        decodedInfo = {}
        for messageId, values in info.items():
            structuredMap, unstructuredList = self._parseFetchPairs(values[0])
            decodedInfo.setdefault(messageId, [[]])[0].extend(unstructuredList)
            results.setdefault(messageId, {}).update(structuredMap)
        info = decodedInfo
        flagChanges = {}
        for messageId in list(results.keys()):
            values = results[messageId]
            for part in list(values.keys()):
                if part not in requestedParts and part == 'FLAGS':
                    flagChanges[messageId] = values['FLAGS']
                    for i in range(len(info[messageId][0])):
                        if info[messageId][0][i] == 'FLAGS':
                            del info[messageId][0][i:i + 2]
                            break
                    del values['FLAGS']
                    if not values:
                        del results[messageId]
        if flagChanges:
            self.flagsChanged(flagChanges)
        if structured:
            return results
        else:
            return info

    def fetchSpecific(self, messages, uid=0, headerType=None, headerNumber=None, headerArgs=None, peek=None, offset=None, length=None):
        """
        Retrieve a specific section of one or more messages

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
            numbers or of unique message IDs.

        @type headerType: L{str}
        @param headerType: If specified, must be one of HEADER, HEADER.FIELDS,
            HEADER.FIELDS.NOT, MIME, or TEXT, and will determine which part of
            the message is retrieved.  For HEADER.FIELDS and HEADER.FIELDS.NOT,
            C{headerArgs} must be a sequence of header names.  For MIME,
            C{headerNumber} must be specified.

        @type headerNumber: L{int} or L{int} sequence
        @param headerNumber: The nested rfc822 index specifying the entity to
            retrieve.  For example, C{1} retrieves the first entity of the
            message, and C{(2, 1, 3}) retrieves the 3rd entity inside the first
            entity inside the second entity of the message.

        @type headerArgs: A sequence of L{str}
        @param headerArgs: If C{headerType} is HEADER.FIELDS, these are the
            headers to retrieve.  If it is HEADER.FIELDS.NOT, these are the
            headers to exclude from retrieval.

        @type peek: C{bool}
        @param peek: If true, cause the server to not set the \\Seen flag on
            this message as a result of this command.

        @type offset: L{int}
        @param offset: The number of octets at the beginning of the result to
            skip.

        @type length: L{int}
        @param length: The number of octets to retrieve.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a mapping of message
            numbers to retrieved data, or whose errback is invoked if there is
            an error.
        """
        fmt = '%s BODY%s[%s%s%s]%s'
        if headerNumber is None:
            number = ''
        elif isinstance(headerNumber, int):
            number = str(headerNumber)
        else:
            number = '.'.join(map(str, headerNumber))
        if headerType is None:
            header = ''
        elif number:
            header = '.' + headerType
        else:
            header = headerType
        if header and headerType in ('HEADER.FIELDS', 'HEADER.FIELDS.NOT'):
            if headerArgs is not None:
                payload = ' (%s)' % ' '.join(headerArgs)
            else:
                payload = ' ()'
        else:
            payload = ''
        if offset is None:
            extra = ''
        else:
            extra = '<%d.%d>' % (offset, length)
        fetch = uid and b'UID FETCH' or b'FETCH'
        cmd = fmt % (messages, peek and '.PEEK' or '', number, header, payload, extra)
        cmd = cmd.encode('charmap')
        d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
        d.addCallback(self._cbFetch, (), False)
        return d

    def _fetch(self, messages, useUID=0, **terms):
        messages = str(messages).encode('ascii')
        fetch = useUID and b'UID FETCH' or b'FETCH'
        if 'rfc822text' in terms:
            del terms['rfc822text']
            terms['rfc822.text'] = True
        if 'rfc822size' in terms:
            del terms['rfc822size']
            terms['rfc822.size'] = True
        if 'rfc822header' in terms:
            del terms['rfc822header']
            terms['rfc822.header'] = True
        encodedTerms = [networkString(s) for s in terms]
        cmd = messages + b' (' + b' '.join([s.upper() for s in encodedTerms]) + b')'
        d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
        d.addCallback(self._cbFetch, [t.upper() for t in terms.keys()], True)
        return d

    def setFlags(self, messages, flags, silent=1, uid=0):
        """
        Set the flags for one or more messages.

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type flags: Any iterable of L{str}
        @param flags: The flags to set

        @type silent: L{bool}
        @param silent: If true, cause the server to suppress its verbose
        response.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of the
        server's responses (C{[]} if C{silent} is true) or whose
        errback is invoked if there is an error.
        """
        return self._store(messages, b'FLAGS', silent, flags, uid)

    def addFlags(self, messages, flags, silent=1, uid=0):
        """
        Add to the set flags for one or more messages.

        This command is allowed in the Selected state.

        @type messages: C{MessageSet} or L{str}
        @param messages: A message sequence set

        @type flags: Any iterable of L{str}
        @param flags: The flags to set

        @type silent: C{bool}
        @param silent: If true, cause the server to suppress its verbose
        response.

        @type uid: C{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of the
        server's responses (C{[]} if C{silent} is true) or whose
        errback is invoked if there is an error.
        """
        return self._store(messages, b'+FLAGS', silent, flags, uid)

    def removeFlags(self, messages, flags, silent=1, uid=0):
        """
        Remove from the set flags for one or more messages.

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type flags: Any iterable of L{str}
        @param flags: The flags to set

        @type silent: L{bool}
        @param silent: If true, cause the server to suppress its verbose
        response.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of the
        server's responses (C{[]} if C{silent} is true) or whose
        errback is invoked if there is an error.
        """
        return self._store(messages, b'-FLAGS', silent, flags, uid)

    def _store(self, messages, cmd, silent, flags, uid):
        messages = str(messages).encode('ascii')
        encodedFlags = [networkString(flag) for flag in flags]
        if silent:
            cmd = cmd + b'.SILENT'
        store = uid and b'UID STORE' or b'STORE'
        args = b' '.join((messages, cmd, b'(' + b' '.join(encodedFlags) + b')'))
        d = self.sendCommand(Command(store, args, wantResponse=(b'FETCH',)))
        expected = ()
        if not silent:
            expected = ('FLAGS',)
        d.addCallback(self._cbFetch, expected, True)
        return d

    def copy(self, messages, mailbox, uid):
        """
        Copy the specified messages to the specified mailbox.

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type mailbox: L{str}
        @param mailbox: The mailbox to which to copy the messages

        @type uid: C{bool}
        @param uid: If true, the C{messages} refers to message UIDs, rather
        than message sequence numbers.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a true value
        when the copy is successful, or whose errback is invoked if there
        is an error.
        """
        messages = str(messages).encode('ascii')
        if uid:
            cmd = b'UID COPY'
        else:
            cmd = b'COPY'
        args = b' '.join([messages, _prepareMailboxName(mailbox)])
        return self.sendCommand(Command(cmd, args))

    def modeChanged(self, writeable):
        """Override me"""

    def flagsChanged(self, newFlags):
        """Override me"""

    def newMessages(self, exists, recent):
        """Override me"""