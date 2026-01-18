import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
class POP3Client(basic.LineOnlyReceiver):
    """
    A POP3 client protocol.

    @type mode: L{int}
    @ivar mode: The type of response expected from the server.  Choices include
    none (0), a one line response (1), the first line of a multi-line
    response (2), and subsequent lines of a multi-line response (3).

    @type command: L{bytes}
    @ivar command: The command most recently sent to the server.

    @type welcomeRe: L{Pattern <re.Pattern.search>}
    @ivar welcomeRe: A regular expression which matches the APOP challenge in
        the server greeting.

    @type welcomeCode: L{bytes}
    @ivar welcomeCode: The APOP challenge passed in the server greeting.
    """
    mode = SHORT
    command = b'WELCOME'
    import re
    welcomeRe = re.compile(b'<(.*)>')

    def __init__(self):
        """
        Issue deprecation warning.
        """
        import warnings
        warnings.warn('twisted.mail.pop3.POP3Client is deprecated, please use twisted.mail.pop3.AdvancedPOP3Client instead.', DeprecationWarning, stacklevel=3)

    def sendShort(self, command, params=None):
        """
        Send a POP3 command to which a short response is expected.

        @type command: L{bytes}
        @param command: A POP3 command.

        @type params: stringifyable L{object} or L{None}
        @param params: Command arguments.
        """
        if params is not None:
            if not isinstance(params, bytes):
                params = str(params).encode('utf-8')
            self.sendLine(command + b' ' + params)
        else:
            self.sendLine(command)
        self.command = command
        self.mode = SHORT

    def sendLong(self, command, params):
        """
        Send a POP3 command to which a long response is expected.

        @type command: L{bytes}
        @param command: A POP3 command.

        @type params: stringifyable L{object}
        @param params: Command arguments.
        """
        if params:
            if not isinstance(params, bytes):
                params = str(params).encode('utf-8')
            self.sendLine(command + b' ' + params)
        else:
            self.sendLine(command)
        self.command = command
        self.mode = FIRST_LONG

    def handle_default(self, line):
        """
        Handle responses from the server for which no other handler exists.

        @type line: L{bytes}
        @param line: A received line.
        """
        if line[:-4] == b'-ERR':
            self.mode = NONE

    def handle_WELCOME(self, line):
        """
        Handle a server response which is expected to be a server greeting.

        @type line: L{bytes}
        @param line: A received line.
        """
        code, data = line.split(b' ', 1)
        if code != b'+OK':
            self.transport.loseConnection()
        else:
            m = self.welcomeRe.match(line)
            if m:
                self.welcomeCode = m.group(1)

    def _dispatch(self, command, default, *args):
        """
        Dispatch a response from the server for handling.

        Command X is dispatched to handle_X() if it exists.  If not, it is
        dispatched to the default handler.

        @type command: L{bytes}
        @param command: The command.

        @type default: callable that takes L{bytes} or
            L{None}
        @param default: The default handler.

        @type args: L{tuple} or L{None}
        @param args: Arguments to the handler function.
        """
        try:
            method = getattr(self, 'handle_' + command.decode('utf-8'), default)
            if method is not None:
                method(*args)
        except BaseException:
            log.err()

    def lineReceived(self, line):
        """
        Dispatch a received line for processing.

        The choice of function to handle the received line is based on the
        type of response expected to the command sent to the server and how
        much of that response has been received.

        An expected one line response to command X is handled by handle_X().
        The first line of a multi-line response to command X is also handled by
        handle_X().  Subsequent lines of the multi-line response are handled by
        handle_X_continue() except for the last line which is handled by
        handle_X_end().

        @type line: L{bytes}
        @param line: A received line.
        """
        if self.mode == SHORT or self.mode == FIRST_LONG:
            self.mode = NEXT[self.mode]
            self._dispatch(self.command, self.handle_default, line)
        elif self.mode == LONG:
            if line == b'.':
                self.mode = NEXT[self.mode]
                self._dispatch(self.command + b'_end', None)
                return
            if line[:1] == b'.':
                line = line[1:]
            self._dispatch(self.command + b'_continue', None, line)

    def apopAuthenticate(self, user, password, magic):
        """
        Perform an authenticated login.

        @type user: L{bytes}
        @param user: The username with which to log in.

        @type password: L{bytes}
        @param password: The password with which to log in.

        @type magic: L{bytes}
        @param magic: The challenge provided by the server.
        """
        digest = md5(magic + password).hexdigest().encode('ascii')
        self.apop(user, digest)

    def apop(self, user, digest):
        """
        Send an APOP command to perform authenticated login.

        @type user: L{bytes}
        @param user: The username with which to log in.

        @type digest: L{bytes}
        @param digest: The challenge response with which to authenticate.
        """
        self.sendLong(b'APOP', b' '.join((user, digest)))

    def retr(self, i):
        """
        Send a RETR command to retrieve a message from the server.

        @type i: L{int} or L{bytes}
        @param i: A 0-based message index.
        """
        self.sendLong(b'RETR', i)

    def dele(self, i):
        """
        Send a DELE command to delete a message from the server.

        @type i: L{int} or L{bytes}
        @param i: A 0-based message index.
        """
        self.sendShort(b'DELE', i)

    def list(self, i=''):
        """
        Send a LIST command to retrieve the size of a message or, if no message
        is specified, the sizes of all messages.

        @type i: L{int} or L{bytes}
        @param i: A 0-based message index or the empty string to specify all
            messages.
        """
        self.sendLong(b'LIST', i)

    def uidl(self, i=''):
        """
        Send a UIDL command to retrieve the unique identifier of a message or,
        if no message is specified, the unique identifiers of all messages.

        @type i: L{int} or L{bytes}
        @param i: A 0-based message index or the empty string to specify all
            messages.
        """
        self.sendLong(b'UIDL', i)

    def user(self, name):
        """
        Send a USER command to perform the first half of a plaintext login.

        @type name: L{bytes}
        @param name: The username with which to log in.
        """
        self.sendShort(b'USER', name)

    def password(self, password):
        """
        Perform the second half of a plaintext login.

        @type password: L{bytes}
        @param password: The plaintext password with which to authenticate.
        """
        self.sendShort(b'PASS', password)
    pass_ = password

    def quit(self):
        """
        Send a QUIT command to disconnect from the server.
        """
        self.sendShort(b'QUIT')