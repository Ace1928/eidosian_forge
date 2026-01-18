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
class IMAP4Server(basic.LineReceiver, policies.TimeoutMixin):
    """
    Protocol implementation for an IMAP4rev1 server.

    The server can be in any of four states:
        - Non-authenticated
        - Authenticated
        - Selected
        - Logout
    """
    IDENT = b'Twisted IMAP4rev1 Ready'
    timeOut = 60
    POSTAUTH_TIMEOUT = 60 * 30
    startedTLS = False
    canStartTLS = False
    tags = None
    portal = None
    account = None
    _onLogout = None
    mbox = None
    _pendingLiteral = None
    _literalStringLimit = 4096
    challengers = None
    _requiresLastMessageInfo = {b'OR', b'NOT', b'UID'}
    state = 'unauth'
    parseState = 'command'

    def __init__(self, chal=None, contextFactory=None, scheduler=None):
        if chal is None:
            chal = {}
        self.challengers = chal
        self.ctx = contextFactory
        if scheduler is None:
            scheduler = iterateInReactor
        self._scheduler = scheduler
        self._queuedAsync = []

    def capabilities(self):
        cap = {b'AUTH': list(self.challengers.keys())}
        if self.ctx and self.canStartTLS:
            if not self.startedTLS and interfaces.ISSLTransport(self.transport, None) is None:
                cap[b'LOGINDISABLED'] = None
                cap[b'STARTTLS'] = None
        cap[b'NAMESPACE'] = None
        cap[b'IDLE'] = None
        return cap

    def connectionMade(self):
        self.tags = {}
        self.canStartTLS = interfaces.ITLSTransport(self.transport, None) is not None
        self.setTimeout(self.timeOut)
        self.sendServerGreeting()

    def connectionLost(self, reason):
        self.setTimeout(None)
        if self._onLogout:
            self._onLogout()
            self._onLogout = None

    def timeoutConnection(self):
        self.sendLine(b'* BYE Autologout; connection idle too long')
        self.transport.loseConnection()
        if self.mbox:
            self.mbox.removeListener(self)
            cmbx = ICloseableMailbox(self.mbox, None)
            if cmbx is not None:
                maybeDeferred(cmbx.close).addErrback(log.err)
            self.mbox = None
        self.state = 'timeout'

    def rawDataReceived(self, data):
        self.resetTimeout()
        passon = self._pendingLiteral.write(data)
        if passon is not None:
            self.setLineMode(passon)
    blocked = None

    def _unblock(self):
        commands = self.blocked
        self.blocked = None
        while commands and self.blocked is None:
            self.lineReceived(commands.pop(0))
        if self.blocked is not None:
            self.blocked.extend(commands)

    def lineReceived(self, line):
        if self.blocked is not None:
            self.blocked.append(line)
            return
        self.resetTimeout()
        f = getattr(self, 'parse_' + self.parseState)
        try:
            f(line)
        except Exception as e:
            self.sendUntaggedResponse(b'BAD Server error: ' + networkString(str(e)))
            log.err()

    def parse_command(self, line):
        args = line.split(None, 2)
        rest = None
        if len(args) == 3:
            tag, cmd, rest = args
        elif len(args) == 2:
            tag, cmd = args
        elif len(args) == 1:
            tag = args[0]
            self.sendBadResponse(tag, b'Missing command')
            return None
        else:
            self.sendBadResponse(None, b'Null command')
            return None
        cmd = cmd.upper()
        try:
            return self.dispatchCommand(tag, cmd, rest)
        except IllegalClientResponse as e:
            self.sendBadResponse(tag, b'Illegal syntax: ' + networkString(str(e)))
        except IllegalOperation as e:
            self.sendNegativeResponse(tag, b'Illegal operation: ' + networkString(str(e)))
        except IllegalMailboxEncoding as e:
            self.sendNegativeResponse(tag, b'Illegal mailbox name: ' + networkString(str(e)))

    def parse_pending(self, line):
        d = self._pendingLiteral
        self._pendingLiteral = None
        self.parseState = 'command'
        d.callback(line)

    def dispatchCommand(self, tag, cmd, rest, uid=None):
        f = self.lookupCommand(cmd)
        if f:
            fn = f[0]
            parseargs = f[1:]
            self.__doCommand(tag, fn, [self, tag], parseargs, rest, uid)
        else:
            self.sendBadResponse(tag, b'Unsupported command')

    def lookupCommand(self, cmd):
        return getattr(self, '_'.join((self.state, nativeString(cmd.upper()))), None)

    def __doCommand(self, tag, handler, args, parseargs, line, uid):
        for i, arg in enumerate(parseargs):
            if callable(arg):
                parseargs = parseargs[i + 1:]
                maybeDeferred(arg, self, line).addCallback(self.__cbDispatch, tag, handler, args, parseargs, uid).addErrback(self.__ebDispatch, tag)
                return
            else:
                args.append(arg)
        if line:
            raise IllegalClientResponse('Too many arguments for command: ' + repr(line))
        if uid is not None:
            handler(*args, uid=uid)
        else:
            handler(*args)

    def __cbDispatch(self, result, tag, fn, args, parseargs, uid):
        arg, rest = result
        args.append(arg)
        self.__doCommand(tag, fn, args, parseargs, rest, uid)

    def __ebDispatch(self, failure, tag):
        if failure.check(IllegalClientResponse):
            self.sendBadResponse(tag, b'Illegal syntax: ' + networkString(str(failure.value)))
        elif failure.check(IllegalOperation):
            self.sendNegativeResponse(tag, b'Illegal operation: ' + networkString(str(failure.value)))
        elif failure.check(IllegalMailboxEncoding):
            self.sendNegativeResponse(tag, b'Illegal mailbox name: ' + networkString(str(failure.value)))
        else:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))
            log.err(failure)

    def _stringLiteral(self, size):
        if size > self._literalStringLimit:
            raise IllegalClientResponse('Literal too long! I accept at most %d octets' % (self._literalStringLimit,))
        d = defer.Deferred()
        self.parseState = 'pending'
        self._pendingLiteral = LiteralString(size, d)
        self.sendContinuationRequest(networkString('Ready for %d octets of text' % size))
        self.setRawMode()
        return d

    def _fileLiteral(self, size):
        d = defer.Deferred()
        self.parseState = 'pending'
        self._pendingLiteral = LiteralFile(size, d)
        self.sendContinuationRequest(networkString('Ready for %d octets of data' % size))
        self.setRawMode()
        return d

    def arg_finalastring(self, line):
        """
        Parse an astring from line that represents a command's final
        argument.  This special case exists to enable parsing empty
        string literals.

        @param line: A line that contains a string literal.
        @type line: L{bytes}

        @return: A 2-tuple containing the parsed argument and any
            trailing data, or a L{Deferred} that fires with that
            2-tuple
        @rtype: L{tuple} of (L{bytes}, L{bytes}) or a L{Deferred}

        @see: https://twistedmatrix.com/trac/ticket/9207
        """
        return self.arg_astring(line, final=True)

    def arg_astring(self, line, final=False):
        """
        Parse an astring from the line, return (arg, rest), possibly
        via a deferred (to handle literals)

        @param line: A line that contains a string literal.
        @type line: L{bytes}

        @param final: Is this the final argument?
        @type final L{bool}

        @return: A 2-tuple containing the parsed argument and any
            trailing data, or a L{Deferred} that fires with that
            2-tuple
        @rtype: L{tuple} of (L{bytes}, L{bytes}) or a L{Deferred}

        """
        line = line.strip()
        if not line:
            raise IllegalClientResponse('Missing argument')
        d = None
        arg, rest = (None, None)
        if line[0:1] == b'"':
            try:
                spam, arg, rest = line.split(b'"', 2)
                rest = rest[1:]
            except ValueError:
                raise IllegalClientResponse('Unmatched quotes')
        elif line[0:1] == b'{':
            if line[-1:] != b'}':
                raise IllegalClientResponse('Malformed literal')
            try:
                size = int(line[1:-1])
            except ValueError:
                raise IllegalClientResponse('Bad literal size: ' + repr(line[1:-1]))
            if final and (not size):
                return (b'', b'')
            d = self._stringLiteral(size)
        else:
            arg = line.split(b' ', 1)
            if len(arg) == 1:
                arg.append(b'')
            arg, rest = arg
        return d or (arg, rest)
    atomre = re.compile(b'(?P<atom>[' + re.escape(_atomChars) + b']+)( (?P<rest>.*$)|$)')

    def arg_atom(self, line):
        """
        Parse an atom from the line
        """
        if not line:
            raise IllegalClientResponse('Missing argument')
        m = self.atomre.match(line)
        if m:
            return (m.group('atom'), m.group('rest'))
        else:
            raise IllegalClientResponse('Malformed ATOM')

    def arg_plist(self, line):
        """
        Parse a (non-nested) parenthesised list from the line
        """
        if not line:
            raise IllegalClientResponse('Missing argument')
        if line[:1] != b'(':
            raise IllegalClientResponse('Missing parenthesis')
        i = line.find(b')')
        if i == -1:
            raise IllegalClientResponse('Mismatched parenthesis')
        return (parseNestedParens(line[1:i], 0), line[i + 2:])

    def arg_literal(self, line):
        """
        Parse a literal from the line
        """
        if not line:
            raise IllegalClientResponse('Missing argument')
        if line[:1] != b'{':
            raise IllegalClientResponse('Missing literal')
        if line[-1:] != b'}':
            raise IllegalClientResponse('Malformed literal')
        try:
            size = int(line[1:-1])
        except ValueError:
            raise IllegalClientResponse(f'Bad literal size: {line[1:-1]!r}')
        return self._fileLiteral(size)

    def arg_searchkeys(self, line):
        """
        searchkeys
        """
        query = parseNestedParens(line)
        return (query, b'')

    def arg_seqset(self, line):
        """
        sequence-set
        """
        rest = b''
        arg = line.split(b' ', 1)
        if len(arg) == 2:
            rest = arg[1]
        arg = arg[0]
        try:
            return (parseIdList(arg), rest)
        except IllegalIdentifierError as e:
            raise IllegalClientResponse('Bad message number ' + str(e))

    def arg_fetchatt(self, line):
        """
        fetch-att
        """
        p = _FetchParser()
        p.parseString(line)
        return (p.result, b'')

    def arg_flaglist(self, line):
        """
        Flag part of store-att-flag
        """
        flags = []
        if line[0:1] == b'(':
            if line[-1:] != b')':
                raise IllegalClientResponse('Mismatched parenthesis')
            line = line[1:-1]
        while line:
            m = self.atomre.search(line)
            if not m:
                raise IllegalClientResponse('Malformed flag')
            if line[0:1] == b'\\' and m.start() == 1:
                flags.append(b'\\' + m.group('atom'))
            elif m.start() == 0:
                flags.append(m.group('atom'))
            else:
                raise IllegalClientResponse('Malformed flag')
            line = m.group('rest')
        return (flags, b'')

    def arg_line(self, line):
        """
        Command line of UID command
        """
        return (line, b'')

    def opt_plist(self, line):
        """
        Optional parenthesised list
        """
        if line.startswith(b'('):
            return self.arg_plist(line)
        else:
            return (None, line)

    def opt_datetime(self, line):
        """
        Optional date-time string
        """
        if line.startswith(b'"'):
            try:
                spam, date, rest = line.split(b'"', 2)
            except ValueError:
                raise IllegalClientResponse('Malformed date-time')
            return (date, rest[1:])
        else:
            return (None, line)

    def opt_charset(self, line):
        """
        Optional charset of SEARCH command
        """
        if line[:7].upper() == b'CHARSET':
            arg = line.split(b' ', 2)
            if len(arg) == 1:
                raise IllegalClientResponse('Missing charset identifier')
            if len(arg) == 2:
                arg.append(b'')
            spam, arg, rest = arg
            return (arg, rest)
        else:
            return (None, line)

    def sendServerGreeting(self):
        msg = b'[CAPABILITY ' + b' '.join(self.listCapabilities()) + b'] ' + self.IDENT
        self.sendPositiveResponse(message=msg)

    def sendBadResponse(self, tag=None, message=b''):
        self._respond(b'BAD', tag, message)

    def sendPositiveResponse(self, tag=None, message=b''):
        self._respond(b'OK', tag, message)

    def sendNegativeResponse(self, tag=None, message=b''):
        self._respond(b'NO', tag, message)

    def sendUntaggedResponse(self, message, isAsync=None, **kwargs):
        isAsync = _get_async_param(isAsync, **kwargs)
        if not isAsync or self.blocked is None:
            self._respond(message, None, None)
        else:
            self._queuedAsync.append(message)

    def sendContinuationRequest(self, msg=b'Ready for additional command text'):
        if msg:
            self.sendLine(b'+ ' + msg)
        else:
            self.sendLine(b'+')

    def _respond(self, state, tag, message):
        if state in (b'OK', b'NO', b'BAD') and self._queuedAsync:
            lines = self._queuedAsync
            self._queuedAsync = []
            for msg in lines:
                self._respond(msg, None, None)
        if not tag:
            tag = b'*'
        if message:
            self.sendLine(b' '.join((tag, state, message)))
        else:
            self.sendLine(b' '.join((tag, state)))

    def listCapabilities(self):
        caps = [b'IMAP4rev1']
        for c, v in self.capabilities().items():
            if v is None:
                caps.append(c)
            elif len(v):
                caps.extend([c + b'=' + cap for cap in v])
        return caps

    def do_CAPABILITY(self, tag):
        self.sendUntaggedResponse(b'CAPABILITY ' + b' '.join(self.listCapabilities()))
        self.sendPositiveResponse(tag, b'CAPABILITY completed')
    unauth_CAPABILITY = (do_CAPABILITY,)
    auth_CAPABILITY = unauth_CAPABILITY
    select_CAPABILITY = unauth_CAPABILITY
    logout_CAPABILITY = unauth_CAPABILITY

    def do_LOGOUT(self, tag):
        self.sendUntaggedResponse(b'BYE Nice talking to you')
        self.sendPositiveResponse(tag, b'LOGOUT successful')
        self.transport.loseConnection()
    unauth_LOGOUT = (do_LOGOUT,)
    auth_LOGOUT = unauth_LOGOUT
    select_LOGOUT = unauth_LOGOUT
    logout_LOGOUT = unauth_LOGOUT

    def do_NOOP(self, tag):
        self.sendPositiveResponse(tag, b'NOOP No operation performed')
    unauth_NOOP = (do_NOOP,)
    auth_NOOP = unauth_NOOP
    select_NOOP = unauth_NOOP
    logout_NOOP = unauth_NOOP

    def do_AUTHENTICATE(self, tag, args):
        args = args.upper().strip()
        if args not in self.challengers:
            self.sendNegativeResponse(tag, b'AUTHENTICATE method unsupported')
        else:
            self.authenticate(self.challengers[args](), tag)
    unauth_AUTHENTICATE = (do_AUTHENTICATE, arg_atom)

    def authenticate(self, chal, tag):
        if self.portal is None:
            self.sendNegativeResponse(tag, b'Temporary authentication failure')
            return
        self._setupChallenge(chal, tag)

    def _setupChallenge(self, chal, tag):
        try:
            challenge = chal.getChallenge()
        except Exception as e:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(e)))
        else:
            coded = encodebytes(challenge)[:-1]
            self.parseState = 'pending'
            self._pendingLiteral = defer.Deferred()
            self.sendContinuationRequest(coded)
            self._pendingLiteral.addCallback(self.__cbAuthChunk, chal, tag)
            self._pendingLiteral.addErrback(self.__ebAuthChunk, tag)

    def __cbAuthChunk(self, result, chal, tag):
        try:
            uncoded = decodebytes(result)
        except binascii.Error:
            raise IllegalClientResponse('Malformed Response - not base64')
        chal.setResponse(uncoded)
        if chal.moreChallenges():
            self._setupChallenge(chal, tag)
        else:
            self.portal.login(chal, None, IAccount).addCallbacks(self.__cbAuthResp, self.__ebAuthResp, (tag,), None, (tag,), None)

    def __cbAuthResp(self, result, tag):
        iface, avatar, logout = result
        assert iface is IAccount, 'IAccount is the only supported interface'
        self.account = avatar
        self.state = 'auth'
        self._onLogout = logout
        self.sendPositiveResponse(tag, b'Authentication successful')
        self.setTimeout(self.POSTAUTH_TIMEOUT)

    def __ebAuthResp(self, failure, tag):
        if failure.check(UnauthorizedLogin):
            self.sendNegativeResponse(tag, b'Authentication failed: unauthorized')
        elif failure.check(UnhandledCredentials):
            self.sendNegativeResponse(tag, b'Authentication failed: server misconfigured')
        else:
            self.sendBadResponse(tag, b'Server error: login failed unexpectedly')
            log.err(failure)

    def __ebAuthChunk(self, failure, tag):
        self.sendNegativeResponse(tag, b'Authentication failed: ' + networkString(str(failure.value)))

    def do_STARTTLS(self, tag):
        if self.startedTLS:
            self.sendNegativeResponse(tag, b'TLS already negotiated')
        elif self.ctx and self.canStartTLS:
            self.sendPositiveResponse(tag, b'Begin TLS negotiation now')
            self.transport.startTLS(self.ctx)
            self.startedTLS = True
            self.challengers = self.challengers.copy()
            if b'LOGIN' not in self.challengers:
                self.challengers[b'LOGIN'] = LOGINCredentials
            if b'PLAIN' not in self.challengers:
                self.challengers[b'PLAIN'] = PLAINCredentials
        else:
            self.sendNegativeResponse(tag, b'TLS not available')
    unauth_STARTTLS = (do_STARTTLS,)

    def do_LOGIN(self, tag, user, passwd):
        if b'LOGINDISABLED' in self.capabilities():
            self.sendBadResponse(tag, b'LOGIN is disabled before STARTTLS')
            return
        maybeDeferred(self.authenticateLogin, user, passwd).addCallback(self.__cbLogin, tag).addErrback(self.__ebLogin, tag)
    unauth_LOGIN = (do_LOGIN, arg_astring, arg_finalastring)

    def authenticateLogin(self, user, passwd):
        """
        Lookup the account associated with the given parameters

        Override this method to define the desired authentication behavior.

        The default behavior is to defer authentication to C{self.portal}
        if it is not None, or to deny the login otherwise.

        @type user: L{str}
        @param user: The username to lookup

        @type passwd: L{str}
        @param passwd: The password to login with
        """
        if self.portal:
            return self.portal.login(credentials.UsernamePassword(user, passwd), None, IAccount)
        raise UnauthorizedLogin()

    def __cbLogin(self, result, tag):
        iface, avatar, logout = result
        if iface is not IAccount:
            self.sendBadResponse(tag, b'Server error: login returned unexpected value')
            log.err(f'__cbLogin called with {iface!r}, IAccount expected')
        else:
            self.account = avatar
            self._onLogout = logout
            self.sendPositiveResponse(tag, b'LOGIN succeeded')
            self.state = 'auth'
            self.setTimeout(self.POSTAUTH_TIMEOUT)

    def __ebLogin(self, failure, tag):
        if failure.check(UnauthorizedLogin):
            self.sendNegativeResponse(tag, b'LOGIN failed')
        else:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))
            log.err(failure)

    def do_NAMESPACE(self, tag):
        personal = public = shared = None
        np = INamespacePresenter(self.account, None)
        if np is not None:
            personal = np.getPersonalNamespaces()
            public = np.getSharedNamespaces()
            shared = np.getSharedNamespaces()
        self.sendUntaggedResponse(b'NAMESPACE ' + collapseNestedLists([personal, public, shared]))
        self.sendPositiveResponse(tag, b'NAMESPACE command completed')
    auth_NAMESPACE = (do_NAMESPACE,)
    select_NAMESPACE = auth_NAMESPACE

    def _selectWork(self, tag, name, rw, cmdName):
        if self.mbox:
            self.mbox.removeListener(self)
            cmbx = ICloseableMailbox(self.mbox, None)
            if cmbx is not None:
                maybeDeferred(cmbx.close).addErrback(log.err)
            self.mbox = None
            self.state = 'auth'
        name = _parseMbox(name)
        maybeDeferred(self.account.select, _parseMbox(name), rw).addCallback(self._cbSelectWork, cmdName, tag).addErrback(self._ebSelectWork, cmdName, tag)

    def _ebSelectWork(self, failure, cmdName, tag):
        self.sendBadResponse(tag, cmdName + b' failed: Server error')
        log.err(failure)

    def _cbSelectWork(self, mbox, cmdName, tag):
        if mbox is None:
            self.sendNegativeResponse(tag, b'No such mailbox')
            return
        if '\\noselect' in [s.lower() for s in mbox.getFlags()]:
            self.sendNegativeResponse(tag, 'Mailbox cannot be selected')
            return
        flags = [networkString(flag) for flag in mbox.getFlags()]
        self.sendUntaggedResponse(b'%d EXISTS' % (mbox.getMessageCount(),))
        self.sendUntaggedResponse(b'%d RECENT' % (mbox.getRecentCount(),))
        self.sendUntaggedResponse(b'FLAGS (' + b' '.join(flags) + b')')
        self.sendPositiveResponse(None, b'[UIDVALIDITY %d]' % (mbox.getUIDValidity(),))
        s = mbox.isWriteable() and b'READ-WRITE' or b'READ-ONLY'
        mbox.addListener(self)
        self.sendPositiveResponse(tag, b'[' + s + b'] ' + cmdName + b' successful')
        self.state = 'select'
        self.mbox = mbox
    auth_SELECT = (_selectWork, arg_astring, 1, b'SELECT')
    select_SELECT = auth_SELECT
    auth_EXAMINE = (_selectWork, arg_astring, 0, b'EXAMINE')
    select_EXAMINE = auth_EXAMINE

    def do_IDLE(self, tag):
        self.sendContinuationRequest(None)
        self.parseTag = tag
        self.lastState = self.parseState
        self.parseState = 'idle'

    def parse_idle(self, *args):
        self.parseState = self.lastState
        del self.lastState
        self.sendPositiveResponse(self.parseTag, b'IDLE terminated')
        del self.parseTag
    select_IDLE = (do_IDLE,)
    auth_IDLE = select_IDLE

    def do_CREATE(self, tag, name):
        name = _parseMbox(name)
        try:
            result = self.account.create(name)
        except MailboxException as c:
            self.sendNegativeResponse(tag, networkString(str(c)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while creating mailbox')
            log.err()
        else:
            if result:
                self.sendPositiveResponse(tag, b'Mailbox created')
            else:
                self.sendNegativeResponse(tag, b'Mailbox not created')
    auth_CREATE = (do_CREATE, arg_finalastring)
    select_CREATE = auth_CREATE

    def do_DELETE(self, tag, name):
        name = _parseMbox(name)
        if name.lower() == 'inbox':
            self.sendNegativeResponse(tag, b'You cannot delete the inbox')
            return
        try:
            self.account.delete(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, str(m).encode('imap4-utf-7'))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while deleting mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Mailbox deleted')
    auth_DELETE = (do_DELETE, arg_finalastring)
    select_DELETE = auth_DELETE

    def do_RENAME(self, tag, oldname, newname):
        oldname, newname = (_parseMbox(n) for n in (oldname, newname))
        if oldname.lower() == 'inbox' or newname.lower() == 'inbox':
            self.sendNegativeResponse(tag, b'You cannot rename the inbox, or rename another mailbox to inbox.')
            return
        try:
            self.account.rename(oldname, newname)
        except TypeError:
            self.sendBadResponse(tag, b'Invalid command syntax')
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while renaming mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Mailbox renamed')
    auth_RENAME = (do_RENAME, arg_astring, arg_finalastring)
    select_RENAME = auth_RENAME

    def do_SUBSCRIBE(self, tag, name):
        name = _parseMbox(name)
        try:
            self.account.subscribe(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while subscribing to mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Subscribed')
    auth_SUBSCRIBE = (do_SUBSCRIBE, arg_finalastring)
    select_SUBSCRIBE = auth_SUBSCRIBE

    def do_UNSUBSCRIBE(self, tag, name):
        name = _parseMbox(name)
        try:
            self.account.unsubscribe(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while unsubscribing from mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Unsubscribed')
    auth_UNSUBSCRIBE = (do_UNSUBSCRIBE, arg_finalastring)
    select_UNSUBSCRIBE = auth_UNSUBSCRIBE

    def _listWork(self, tag, ref, mbox, sub, cmdName):
        mbox = _parseMbox(mbox)
        ref = _parseMbox(ref)
        maybeDeferred(self.account.listMailboxes, ref, mbox).addCallback(self._cbListWork, tag, sub, cmdName).addErrback(self._ebListWork, tag)

    def _cbListWork(self, mailboxes, tag, sub, cmdName):
        for name, box in mailboxes:
            if not sub or self.account.isSubscribed(name):
                flags = [networkString(flag) for flag in box.getFlags()]
                delim = box.getHierarchicalDelimiter().encode('imap4-utf-7')
                resp = (DontQuoteMe(cmdName), map(DontQuoteMe, flags), delim, name.encode('imap4-utf-7'))
                self.sendUntaggedResponse(collapseNestedLists(resp))
        self.sendPositiveResponse(tag, cmdName + b' completed')

    def _ebListWork(self, failure, tag):
        self.sendBadResponse(tag, b'Server error encountered while listing mailboxes.')
        log.err(failure)
    auth_LIST = (_listWork, arg_astring, arg_astring, 0, b'LIST')
    select_LIST = auth_LIST
    auth_LSUB = (_listWork, arg_astring, arg_astring, 1, b'LSUB')
    select_LSUB = auth_LSUB

    def do_STATUS(self, tag, mailbox, names):
        nativeNames = []
        for name in names:
            nativeNames.append(nativeString(name))
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox, 0).addCallback(self._cbStatusGotMailbox, tag, mailbox, nativeNames).addErrback(self._ebStatusGotMailbox, tag)

    def _cbStatusGotMailbox(self, mbox, tag, mailbox, names):
        if mbox:
            maybeDeferred(mbox.requestStatus, names).addCallbacks(self.__cbStatus, self.__ebStatus, (tag, mailbox), None, (tag, mailbox), None)
        else:
            self.sendNegativeResponse(tag, b'Could not open mailbox')

    def _ebStatusGotMailbox(self, failure, tag):
        self.sendBadResponse(tag, b'Server error encountered while opening mailbox.')
        log.err(failure)
    auth_STATUS = (do_STATUS, arg_astring, arg_plist)
    select_STATUS = auth_STATUS

    def __cbStatus(self, status, tag, box):
        line = networkString(' '.join(['%s %s' % x for x in status.items()]))
        self.sendUntaggedResponse(b'STATUS ' + box.encode('imap4-utf-7') + b' (' + line + b')')
        self.sendPositiveResponse(tag, b'STATUS complete')

    def __ebStatus(self, failure, tag, box):
        self.sendBadResponse(tag, b'STATUS ' + box + b' failed: ' + networkString(str(failure.value)))

    def do_APPEND(self, tag, mailbox, flags, date, message):
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox).addCallback(self._cbAppendGotMailbox, tag, flags, date, message).addErrback(self._ebAppendGotMailbox, tag)

    def _cbAppendGotMailbox(self, mbox, tag, flags, date, message):
        if not mbox:
            self.sendNegativeResponse(tag, '[TRYCREATE] No such mailbox')
            return
        decodedFlags = [nativeString(flag) for flag in flags]
        d = mbox.addMessage(message, decodedFlags, date)
        d.addCallback(self.__cbAppend, tag, mbox)
        d.addErrback(self.__ebAppend, tag)

    def _ebAppendGotMailbox(self, failure, tag):
        self.sendBadResponse(tag, b'Server error encountered while opening mailbox.')
        log.err(failure)
    auth_APPEND = (do_APPEND, arg_astring, opt_plist, opt_datetime, arg_literal)
    select_APPEND = auth_APPEND

    def __cbAppend(self, result, tag, mbox):
        self.sendUntaggedResponse(b'%d EXISTS' % (mbox.getMessageCount(),))
        self.sendPositiveResponse(tag, b'APPEND complete')

    def __ebAppend(self, failure, tag):
        self.sendBadResponse(tag, b'APPEND failed: ' + networkString(str(failure.value)))

    def do_CHECK(self, tag):
        d = self.checkpoint()
        if d is None:
            self.__cbCheck(None, tag)
        else:
            d.addCallbacks(self.__cbCheck, self.__ebCheck, callbackArgs=(tag,), errbackArgs=(tag,))
    select_CHECK = (do_CHECK,)

    def __cbCheck(self, result, tag):
        self.sendPositiveResponse(tag, b'CHECK completed')

    def __ebCheck(self, failure, tag):
        self.sendBadResponse(tag, b'CHECK failed: ' + networkString(str(failure.value)))

    def checkpoint(self):
        """
        Called when the client issues a CHECK command.

        This should perform any checkpoint operations required by the server.
        It may be a long running operation, but may not block.  If it returns
        a deferred, the client will only be informed of success (or failure)
        when the deferred's callback (or errback) is invoked.
        """
        return None

    def do_CLOSE(self, tag):
        d = None
        if self.mbox.isWriteable():
            d = maybeDeferred(self.mbox.expunge)
        cmbx = ICloseableMailbox(self.mbox, None)
        if cmbx is not None:
            if d is not None:
                d.addCallback(lambda result: cmbx.close())
            else:
                d = maybeDeferred(cmbx.close)
        if d is not None:
            d.addCallbacks(self.__cbClose, self.__ebClose, (tag,), None, (tag,), None)
        else:
            self.__cbClose(None, tag)
    select_CLOSE = (do_CLOSE,)

    def __cbClose(self, result, tag):
        self.sendPositiveResponse(tag, b'CLOSE completed')
        self.mbox.removeListener(self)
        self.mbox = None
        self.state = 'auth'

    def __ebClose(self, failure, tag):
        self.sendBadResponse(tag, b'CLOSE failed: ' + networkString(str(failure.value)))

    def do_EXPUNGE(self, tag):
        if self.mbox.isWriteable():
            maybeDeferred(self.mbox.expunge).addCallbacks(self.__cbExpunge, self.__ebExpunge, (tag,), None, (tag,), None)
        else:
            self.sendNegativeResponse(tag, b'EXPUNGE ignored on read-only mailbox')
    select_EXPUNGE = (do_EXPUNGE,)

    def __cbExpunge(self, result, tag):
        for e in result:
            self.sendUntaggedResponse(b'%d EXPUNGE' % (e,))
        self.sendPositiveResponse(tag, b'EXPUNGE completed')

    def __ebExpunge(self, failure, tag):
        self.sendBadResponse(tag, b'EXPUNGE failed: ' + networkString(str(failure.value)))
        log.err(failure)

    def do_SEARCH(self, tag, charset, query, uid=0):
        sm = ISearchableMailbox(self.mbox, None)
        if sm is not None:
            maybeDeferred(sm.search, query, uid=uid).addCallback(self.__cbSearch, tag, self.mbox, uid).addErrback(self.__ebSearch, tag)
        else:
            s = parseIdList(b'1:*')
            maybeDeferred(self.mbox.fetch, s, uid=uid).addCallback(self.__cbManualSearch, tag, self.mbox, query, uid).addErrback(self.__ebSearch, tag)
    select_SEARCH = (do_SEARCH, opt_charset, arg_searchkeys)

    def __cbSearch(self, result, tag, mbox, uid):
        if uid:
            result = map(mbox.getUID, result)
        ids = networkString(' '.join([str(i) for i in result]))
        self.sendUntaggedResponse(b'SEARCH ' + ids)
        self.sendPositiveResponse(tag, b'SEARCH completed')

    def __cbManualSearch(self, result, tag, mbox, query, uid, searchResults=None):
        """
        Apply the search filter to a set of messages. Send the response to the
        client.

        @type result: L{list} of L{tuple} of (L{int}, provider of
            L{imap4.IMessage})
        @param result: A list two tuples of messages with their sequence ids,
            sorted by the ids in descending order.

        @type tag: L{str}
        @param tag: A command tag.

        @type mbox: Provider of L{imap4.IMailbox}
        @param mbox: The searched mailbox.

        @type query: L{list}
        @param query: A list representing the parsed form of the search query.

        @param uid: A flag indicating whether the search is over message
            sequence numbers or UIDs.

        @type searchResults: L{list}
        @param searchResults: The search results so far or L{None} if no
            results yet.
        """
        if searchResults is None:
            searchResults = []
        i = 0
        lastSequenceId = result and result[-1][0]
        lastMessageId = result and result[-1][1].getUID()
        for i, (msgId, msg) in list(zip(range(5), result)):
            if self._searchFilter(copy.deepcopy(query), msgId, msg, lastSequenceId, lastMessageId):
                searchResults.append(b'%d' % (msg.getUID() if uid else msgId,))
        if i == 4:
            from twisted.internet import reactor
            reactor.callLater(0, self.__cbManualSearch, list(result[5:]), tag, mbox, query, uid, searchResults)
        else:
            if searchResults:
                self.sendUntaggedResponse(b'SEARCH ' + b' '.join(searchResults))
            self.sendPositiveResponse(tag, b'SEARCH completed')

    def _searchFilter(self, query, id, msg, lastSequenceId, lastMessageId):
        """
        Pop search terms from the beginning of C{query} until there are none
        left and apply them to the given message.

        @param query: A list representing the parsed form of the search query.

        @param id: The sequence number of the message being checked.

        @param msg: The message being checked.

        @type lastSequenceId: L{int}
        @param lastSequenceId: The highest sequence number of any message in
            the mailbox being searched.

        @type lastMessageId: L{int}
        @param lastMessageId: The highest UID of any message in the mailbox
            being searched.

        @return: Boolean indicating whether all of the query terms match the
            message.
        """
        while query:
            if not self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId):
                return False
        return True

    def _singleSearchStep(self, query, msgId, msg, lastSequenceId, lastMessageId):
        """
        Pop one search term from the beginning of C{query} (possibly more than
        one element) and return whether it matches the given message.

        @param query: A list representing the parsed form of the search query.

        @param msgId: The sequence number of the message being checked.

        @param msg: The message being checked.

        @param lastSequenceId: The highest sequence number of any message in
            the mailbox being searched.

        @param lastMessageId: The highest UID of any message in the mailbox
            being searched.

        @return: Boolean indicating whether the query term matched the message.
        """
        q = query.pop(0)
        if isinstance(q, list):
            if not self._searchFilter(q, msgId, msg, lastSequenceId, lastMessageId):
                return False
        else:
            c = q.upper()
            if not c[:1].isalpha():
                messageSet = parseIdList(c, lastSequenceId)
                return msgId in messageSet
            else:
                f = getattr(self, 'search_' + nativeString(c), None)
                if f is None:
                    raise IllegalQueryError('Invalid search command %s' % nativeString(c))
                if c in self._requiresLastMessageInfo:
                    result = f(query, msgId, msg, (lastSequenceId, lastMessageId))
                else:
                    result = f(query, msgId, msg)
                if not result:
                    return False
        return True

    def search_ALL(self, query, id, msg):
        """
        Returns C{True} if the message matches the ALL search key (always).

        @type query: A L{list} of L{str}
        @param query: A list representing the parsed query string.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        """
        return True

    def search_ANSWERED(self, query, id, msg):
        """
        Returns C{True} if the message has been answered.

        @type query: A L{list} of L{str}
        @param query: A list representing the parsed query string.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        """
        return '\\Answered' in msg.getFlags()

    def search_BCC(self, query, id, msg):
        """
        Returns C{True} if the message has a BCC address matching the query.

        @type query: A L{list} of L{str}
        @param query: A list whose first element is a BCC L{str}

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        """
        bcc = msg.getHeaders(False, 'bcc').get('bcc', '')
        return bcc.lower().find(query.pop(0).lower()) != -1

    def search_BEFORE(self, query, id, msg):
        date = parseTime(query.pop(0))
        return email.utils.parsedate(nativeString(msg.getInternalDate())) < date

    def search_BODY(self, query, id, msg):
        body = query.pop(0).lower()
        return text.strFile(body, msg.getBodyFile(), False)

    def search_CC(self, query, id, msg):
        cc = msg.getHeaders(False, 'cc').get('cc', '')
        return cc.lower().find(query.pop(0).lower()) != -1

    def search_DELETED(self, query, id, msg):
        return '\\Deleted' in msg.getFlags()

    def search_DRAFT(self, query, id, msg):
        return '\\Draft' in msg.getFlags()

    def search_FLAGGED(self, query, id, msg):
        return '\\Flagged' in msg.getFlags()

    def search_FROM(self, query, id, msg):
        fm = msg.getHeaders(False, 'from').get('from', '')
        return fm.lower().find(query.pop(0).lower()) != -1

    def search_HEADER(self, query, id, msg):
        hdr = query.pop(0).lower()
        hdr = msg.getHeaders(False, hdr).get(hdr, '')
        return hdr.lower().find(query.pop(0).lower()) != -1

    def search_KEYWORD(self, query, id, msg):
        query.pop(0)
        return False

    def search_LARGER(self, query, id, msg):
        return int(query.pop(0)) < msg.getSize()

    def search_NEW(self, query, id, msg):
        return '\\Recent' in msg.getFlags() and '\\Seen' not in msg.getFlags()

    def search_NOT(self, query, id, msg, lastIDs):
        """
        Returns C{True} if the message does not match the query.

        @type query: A L{list} of L{str}
        @param query: A list representing the parsed form of the search query.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        @param msg: The message being checked.

        @type lastIDs: L{tuple}
        @param lastIDs: A tuple of (last sequence id, last message id).
        The I{last sequence id} is an L{int} containing the highest sequence
        number of a message in the mailbox.  The I{last message id} is an
        L{int} containing the highest UID of a message in the mailbox.
        """
        lastSequenceId, lastMessageId = lastIDs
        return not self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)

    def search_OLD(self, query, id, msg):
        return '\\Recent' not in msg.getFlags()

    def search_ON(self, query, id, msg):
        date = parseTime(query.pop(0))
        return email.utils.parsedate(msg.getInternalDate()) == date

    def search_OR(self, query, id, msg, lastIDs):
        """
        Returns C{True} if the message matches any of the first two query
        items.

        @type query: A L{list} of L{str}
        @param query: A list representing the parsed form of the search query.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        @param msg: The message being checked.

        @type lastIDs: L{tuple}
        @param lastIDs: A tuple of (last sequence id, last message id).
        The I{last sequence id} is an L{int} containing the highest sequence
        number of a message in the mailbox.  The I{last message id} is an
        L{int} containing the highest UID of a message in the mailbox.
        """
        lastSequenceId, lastMessageId = lastIDs
        a = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
        b = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
        return a or b

    def search_RECENT(self, query, id, msg):
        return '\\Recent' in msg.getFlags()

    def search_SEEN(self, query, id, msg):
        return '\\Seen' in msg.getFlags()

    def search_SENTBEFORE(self, query, id, msg):
        """
        Returns C{True} if the message date is earlier than the query date.

        @type query: A L{list} of L{str}
        @param query: A list whose first element starts with a stringified date
            that is a fragment of an L{imap4.Query()}. The date must be in the
            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        """
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date < parseTime(query.pop(0))

    def search_SENTON(self, query, id, msg):
        """
        Returns C{True} if the message date is the same as the query date.

        @type query: A L{list} of L{str}
        @param query: A list whose first element starts with a stringified date
            that is a fragment of an L{imap4.Query()}. The date must be in the
            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.

        @type msg: Provider of L{imap4.IMessage}
        """
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date[:3] == parseTime(query.pop(0))[:3]

    def search_SENTSINCE(self, query, id, msg):
        """
        Returns C{True} if the message date is later than the query date.

        @type query: A L{list} of L{str}
        @param query: A list whose first element starts with a stringified date
            that is a fragment of an L{imap4.Query()}. The date must be in the
            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.

        @type msg: Provider of L{imap4.IMessage}
        """
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date > parseTime(query.pop(0))

    def search_SINCE(self, query, id, msg):
        date = parseTime(query.pop(0))
        return email.utils.parsedate(msg.getInternalDate()) > date

    def search_SMALLER(self, query, id, msg):
        return int(query.pop(0)) > msg.getSize()

    def search_SUBJECT(self, query, id, msg):
        subj = msg.getHeaders(False, 'subject').get('subject', '')
        return subj.lower().find(query.pop(0).lower()) != -1

    def search_TEXT(self, query, id, msg):
        body = query.pop(0).lower()
        return text.strFile(body, msg.getBodyFile(), False)

    def search_TO(self, query, id, msg):
        to = msg.getHeaders(False, 'to').get('to', '')
        return to.lower().find(query.pop(0).lower()) != -1

    def search_UID(self, query, id, msg, lastIDs):
        """
        Returns C{True} if the message UID is in the range defined by the
        search query.

        @type query: A L{list} of L{bytes}
        @param query: A list representing the parsed form of the search
            query. Its first element should be a L{str} that can be interpreted
            as a sequence range, for example '2:4,5:*'.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        @param msg: The message being checked.

        @type lastIDs: L{tuple}
        @param lastIDs: A tuple of (last sequence id, last message id).
        The I{last sequence id} is an L{int} containing the highest sequence
        number of a message in the mailbox.  The I{last message id} is an
        L{int} containing the highest UID of a message in the mailbox.
        """
        lastSequenceId, lastMessageId = lastIDs
        c = query.pop(0)
        m = parseIdList(c, lastMessageId)
        return msg.getUID() in m

    def search_UNANSWERED(self, query, id, msg):
        return '\\Answered' not in msg.getFlags()

    def search_UNDELETED(self, query, id, msg):
        return '\\Deleted' not in msg.getFlags()

    def search_UNDRAFT(self, query, id, msg):
        return '\\Draft' not in msg.getFlags()

    def search_UNFLAGGED(self, query, id, msg):
        return '\\Flagged' not in msg.getFlags()

    def search_UNKEYWORD(self, query, id, msg):
        query.pop(0)
        return False

    def search_UNSEEN(self, query, id, msg):
        return '\\Seen' not in msg.getFlags()

    def __ebSearch(self, failure, tag):
        self.sendBadResponse(tag, b'SEARCH failed: ' + networkString(str(failure.value)))
        log.err(failure)

    def do_FETCH(self, tag, messages, query, uid=0):
        if query:
            self._oldTimeout = self.setTimeout(None)
            maybeDeferred(self.mbox.fetch, messages, uid=uid).addCallback(iter).addCallback(self.__cbFetch, tag, query, uid).addErrback(self.__ebFetch, tag)
        else:
            self.sendPositiveResponse(tag, b'FETCH complete')
    select_FETCH = (do_FETCH, arg_seqset, arg_fetchatt)

    def __cbFetch(self, results, tag, query, uid):
        if self.blocked is None:
            self.blocked = []
        try:
            id, msg = next(results)
        except StopIteration:
            self.setTimeout(self._oldTimeout)
            del self._oldTimeout
            self.sendPositiveResponse(tag, b'FETCH completed')
            self._unblock()
        else:
            self.spewMessage(id, msg, query, uid).addCallback(lambda _: self.__cbFetch(results, tag, query, uid)).addErrback(self.__ebSpewMessage)

    def __ebSpewMessage(self, failure):
        log.err(failure)
        self.transport.loseConnection()

    def spew_envelope(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        _w(b'ENVELOPE ' + collapseNestedLists([getEnvelope(msg)]))

    def spew_flags(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.writen
        encodedFlags = [networkString(flag) for flag in msg.getFlags()]
        _w(b'FLAGS ' + b'(' + b' '.join(encodedFlags) + b')')

    def spew_internaldate(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        idate = msg.getInternalDate()
        ttup = email.utils.parsedate_tz(nativeString(idate))
        if ttup is None:
            log.msg('%d:%r: unpareseable internaldate: %r' % (id, msg, idate))
            raise IMAP4Exception('Internal failure generating INTERNALDATE')
        strdate = time.strftime('%d-%%s-%Y %H:%M:%S ', ttup[:9])
        odate = networkString(strdate % (_MONTH_NAMES[ttup[1]],))
        if ttup[9] is None:
            odate = odate + b'+0000'
        else:
            if ttup[9] >= 0:
                sign = b'+'
            else:
                sign = b'-'
            odate = odate + sign + b'%04d' % (abs(ttup[9]) // 3600 * 100 + abs(ttup[9]) % 3600 // 60,)
        _w(b'INTERNALDATE ' + _quote(odate))

    def spew_rfc822header(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        hdrs = _formatHeaders(msg.getHeaders(True))
        _w(b'RFC822.HEADER ' + _literal(hdrs))

    def spew_rfc822text(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822.TEXT ')
        _f()
        return FileProducer(msg.getBodyFile()).beginProducing(self.transport)

    def spew_rfc822size(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822.SIZE %d' % (msg.getSize(),))

    def spew_rfc822(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822 ')
        _f()
        mf = IMessageFile(msg, None)
        if mf is not None:
            return FileProducer(mf.open()).beginProducing(self.transport)
        return MessageProducer(msg, None, self._scheduler).beginProducing(self.transport)

    def spew_uid(self, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        _w(b'UID %d' % (msg.getUID(),))

    def spew_bodystructure(self, id, msg, _w=None, _f=None):
        _w(b'BODYSTRUCTURE ' + collapseNestedLists([getBodyStructure(msg, True)]))

    def spew_body(self, part, id, msg, _w=None, _f=None):
        if _w is None:
            _w = self.transport.write
        for p in part.part:
            if msg.isMultipart():
                msg = msg.getSubPart(p)
            elif p > 0:
                raise TypeError('Requested subpart of non-multipart message')
        if part.header:
            hdrs = msg.getHeaders(part.header.negate, *part.header.fields)
            hdrs = _formatHeaders(hdrs)
            _w(part.__bytes__() + b' ' + _literal(hdrs))
        elif part.text:
            _w(part.__bytes__() + b' ')
            _f()
            return FileProducer(msg.getBodyFile()).beginProducing(self.transport)
        elif part.mime:
            hdrs = _formatHeaders(msg.getHeaders(True))
            _w(part.__bytes__() + b' ' + _literal(hdrs))
        elif part.empty:
            _w(part.__bytes__() + b' ')
            _f()
            if part.part:
                return FileProducer(msg.getBodyFile()).beginProducing(self.transport)
            else:
                mf = IMessageFile(msg, None)
                if mf is not None:
                    return FileProducer(mf.open()).beginProducing(self.transport)
                return MessageProducer(msg, None, self._scheduler).beginProducing(self.transport)
        else:
            _w(b'BODY ' + collapseNestedLists([getBodyStructure(msg)]))

    def spewMessage(self, id, msg, query, uid):
        wbuf = WriteBuffer(self.transport)
        write = wbuf.write
        flush = wbuf.flush

        def start():
            write(b'* %d FETCH (' % (id,))

        def finish():
            write(b')\r\n')

        def space():
            write(b' ')

        def spew():
            seenUID = False
            start()
            for part in query:
                if part.type == 'uid':
                    seenUID = True
                if part.type == 'body':
                    yield self.spew_body(part, id, msg, write, flush)
                else:
                    f = getattr(self, 'spew_' + part.type)
                    yield f(id, msg, write, flush)
                if part is not query[-1]:
                    space()
            if uid and (not seenUID):
                space()
                yield self.spew_uid(id, msg, write, flush)
            finish()
            flush()
        return self._scheduler(spew())

    def __ebFetch(self, failure, tag):
        self.setTimeout(self._oldTimeout)
        del self._oldTimeout
        log.err(failure)
        self.sendBadResponse(tag, b'FETCH failed: ' + networkString(str(failure.value)))

    def do_STORE(self, tag, messages, mode, flags, uid=0):
        mode = mode.upper()
        silent = mode.endswith(b'SILENT')
        if mode.startswith(b'+'):
            mode = 1
        elif mode.startswith(b'-'):
            mode = -1
        else:
            mode = 0
        flags = [nativeString(flag) for flag in flags]
        maybeDeferred(self.mbox.store, messages, flags, mode, uid=uid).addCallbacks(self.__cbStore, self.__ebStore, (tag, self.mbox, uid, silent), None, (tag,), None)
    select_STORE = (do_STORE, arg_seqset, arg_atom, arg_flaglist)

    def __cbStore(self, result, tag, mbox, uid, silent):
        if result and (not silent):
            for k, v in result.items():
                if uid:
                    uidstr = b' UID %d' % (mbox.getUID(k),)
                else:
                    uidstr = b''
                flags = [networkString(flag) for flag in v]
                self.sendUntaggedResponse(b'%d FETCH (FLAGS (%b)%b)' % (k, b' '.join(flags), uidstr))
        self.sendPositiveResponse(tag, b'STORE completed')

    def __ebStore(self, failure, tag):
        self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))

    def do_COPY(self, tag, messages, mailbox, uid=0):
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox).addCallback(self._cbCopySelectedMailbox, tag, messages, mailbox, uid).addErrback(self._ebCopySelectedMailbox, tag)
    select_COPY = (do_COPY, arg_seqset, arg_finalastring)

    def _cbCopySelectedMailbox(self, mbox, tag, messages, mailbox, uid):
        if not mbox:
            self.sendNegativeResponse(tag, 'No such mailbox: ' + mailbox)
        else:
            maybeDeferred(self.mbox.fetch, messages, uid).addCallback(self.__cbCopy, tag, mbox).addCallback(self.__cbCopied, tag, mbox).addErrback(self.__ebCopy, tag)

    def _ebCopySelectedMailbox(self, failure, tag):
        self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))

    def __cbCopy(self, messages, tag, mbox):
        addedDeferreds = []
        fastCopyMbox = IMessageCopier(mbox, None)
        for id, msg in messages:
            if fastCopyMbox is not None:
                d = maybeDeferred(fastCopyMbox.copy, msg)
                addedDeferreds.append(d)
                continue
            flags = msg.getFlags()
            date = msg.getInternalDate()
            body = IMessageFile(msg, None)
            if body is not None:
                bodyFile = body.open()
                d = maybeDeferred(mbox.addMessage, bodyFile, flags, date)
            else:

                def rewind(f):
                    f.seek(0)
                    return f
                buffer = tempfile.TemporaryFile()
                d = MessageProducer(msg, buffer, self._scheduler).beginProducing(None).addCallback(lambda _, b=buffer, f=flags, d=date: mbox.addMessage(rewind(b), f, d))
            addedDeferreds.append(d)
        return defer.DeferredList(addedDeferreds)

    def __cbCopied(self, deferredIds, tag, mbox):
        ids = []
        failures = []
        for status, result in deferredIds:
            if status:
                ids.append(result)
            else:
                failures.append(result.value)
        if failures:
            self.sendNegativeResponse(tag, '[ALERT] Some messages were not copied')
        else:
            self.sendPositiveResponse(tag, b'COPY completed')

    def __ebCopy(self, failure, tag):
        self.sendBadResponse(tag, b'COPY failed:' + networkString(str(failure.value)))
        log.err(failure)

    def do_UID(self, tag, command, line):
        command = command.upper()
        if command not in (b'COPY', b'FETCH', b'STORE', b'SEARCH'):
            raise IllegalClientResponse(command)
        self.dispatchCommand(tag, command, line, uid=1)
    select_UID = (do_UID, arg_atom, arg_line)

    def modeChanged(self, writeable):
        if writeable:
            self.sendUntaggedResponse(message=b'[READ-WRITE]', isAsync=True)
        else:
            self.sendUntaggedResponse(message=b'[READ-ONLY]', isAsync=True)

    def flagsChanged(self, newFlags):
        for mId, flags in newFlags.items():
            encodedFlags = [networkString(flag) for flag in flags]
            msg = b'%d FETCH (FLAGS (%b))' % (mId, b' '.join(encodedFlags))
            self.sendUntaggedResponse(msg, isAsync=True)

    def newMessages(self, exists, recent):
        if exists is not None:
            self.sendUntaggedResponse(b'%d EXISTS' % (exists,), isAsync=True)
        if recent is not None:
            self.sendUntaggedResponse(b'%d RECENT' % (recent,), isAsync=True)