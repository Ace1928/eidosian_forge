import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
class NNTP:
    encoding = 'utf-8'
    errors = 'surrogateescape'

    def __init__(self, host, port=NNTP_PORT, user=None, password=None, readermode=None, usenetrc=False, timeout=_GLOBAL_DEFAULT_TIMEOUT):
        """Initialize an instance.  Arguments:
        - host: hostname to connect to
        - port: port to connect to (default the standard NNTP port)
        - user: username to authenticate with
        - password: password to use with username
        - readermode: if true, send 'mode reader' command after
                      connecting.
        - usenetrc: allow loading username and password from ~/.netrc file
                    if not specified explicitly
        - timeout: timeout (in seconds) used for socket connections

        readermode is sometimes necessary if you are connecting to an
        NNTP server on the local machine and intend to call
        reader-specific commands, such as `group'.  If you get
        unexpected NNTPPermanentErrors, you might need to set
        readermode.
        """
        self.host = host
        self.port = port
        self.sock = self._create_socket(timeout)
        self.file = None
        try:
            self.file = self.sock.makefile('rwb')
            self._base_init(readermode)
            if user or usenetrc:
                self.login(user, password, usenetrc)
        except:
            if self.file:
                self.file.close()
            self.sock.close()
            raise

    def _base_init(self, readermode):
        """Partial initialization for the NNTP protocol.
        This instance method is extracted for supporting the test code.
        """
        self.debugging = 0
        self.welcome = self._getresp()
        self._caps = None
        self.getcapabilities()
        self.readermode_afterauth = False
        if readermode and 'READER' not in self._caps:
            self._setreadermode()
            if not self.readermode_afterauth:
                self._caps = None
                self.getcapabilities()
        self.tls_on = False
        self.authenticated = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        is_connected = lambda: hasattr(self, 'file')
        if is_connected():
            try:
                self.quit()
            except (OSError, EOFError):
                pass
            finally:
                if is_connected():
                    self._close()

    def _create_socket(self, timeout):
        if timeout is not None and (not timeout):
            raise ValueError('Non-blocking socket (timeout=0) is not supported')
        sys.audit('nntplib.connect', self, self.host, self.port)
        return socket.create_connection((self.host, self.port), timeout)

    def getwelcome(self):
        """Get the welcome message from the server
        (this is read and squirreled away by __init__()).
        If the response code is 200, posting is allowed;
        if it 201, posting is not allowed."""
        if self.debugging:
            print('*welcome*', repr(self.welcome))
        return self.welcome

    def getcapabilities(self):
        """Get the server capabilities, as read by __init__().
        If the CAPABILITIES command is not supported, an empty dict is
        returned."""
        if self._caps is None:
            self.nntp_version = 1
            self.nntp_implementation = None
            try:
                resp, caps = self.capabilities()
            except (NNTPPermanentError, NNTPTemporaryError):
                self._caps = {}
            else:
                self._caps = caps
                if 'VERSION' in caps:
                    self.nntp_version = max(map(int, caps['VERSION']))
                if 'IMPLEMENTATION' in caps:
                    self.nntp_implementation = ' '.join(caps['IMPLEMENTATION'])
        return self._caps

    def set_debuglevel(self, level):
        """Set the debugging level.  Argument 'level' means:
        0: no debugging output (default)
        1: print commands and responses but not body text etc.
        2: also print raw lines read and sent before stripping CR/LF"""
        self.debugging = level
    debug = set_debuglevel

    def _putline(self, line):
        """Internal: send one line to the server, appending CRLF.
        The `line` must be a bytes-like object."""
        sys.audit('nntplib.putline', self, line)
        line = line + _CRLF
        if self.debugging > 1:
            print('*put*', repr(line))
        self.file.write(line)
        self.file.flush()

    def _putcmd(self, line):
        """Internal: send one command to the server (through _putline()).
        The `line` must be a unicode string."""
        if self.debugging:
            print('*cmd*', repr(line))
        line = line.encode(self.encoding, self.errors)
        self._putline(line)

    def _getline(self, strip_crlf=True):
        """Internal: return one line from the server, stripping _CRLF.
        Raise EOFError if the connection is closed.
        Returns a bytes object."""
        line = self.file.readline(_MAXLINE + 1)
        if len(line) > _MAXLINE:
            raise NNTPDataError('line too long')
        if self.debugging > 1:
            print('*get*', repr(line))
        if not line:
            raise EOFError
        if strip_crlf:
            if line[-2:] == _CRLF:
                line = line[:-2]
            elif line[-1:] in _CRLF:
                line = line[:-1]
        return line

    def _getresp(self):
        """Internal: get a response from the server.
        Raise various errors if the response indicates an error.
        Returns a unicode string."""
        resp = self._getline()
        if self.debugging:
            print('*resp*', repr(resp))
        resp = resp.decode(self.encoding, self.errors)
        c = resp[:1]
        if c == '4':
            raise NNTPTemporaryError(resp)
        if c == '5':
            raise NNTPPermanentError(resp)
        if c not in '123':
            raise NNTPProtocolError(resp)
        return resp

    def _getlongresp(self, file=None):
        """Internal: get a response plus following text from the server.
        Raise various errors if the response indicates an error.

        Returns a (response, lines) tuple where `response` is a unicode
        string and `lines` is a list of bytes objects.
        If `file` is a file-like object, it must be open in binary mode.
        """
        openedFile = None
        try:
            if isinstance(file, (str, bytes)):
                openedFile = file = open(file, 'wb')
            resp = self._getresp()
            if resp[:3] not in _LONGRESP:
                raise NNTPReplyError(resp)
            lines = []
            if file is not None:
                terminators = (b'.' + _CRLF, b'.\n')
                while 1:
                    line = self._getline(False)
                    if line in terminators:
                        break
                    if line.startswith(b'..'):
                        line = line[1:]
                    file.write(line)
            else:
                terminator = b'.'
                while 1:
                    line = self._getline()
                    if line == terminator:
                        break
                    if line.startswith(b'..'):
                        line = line[1:]
                    lines.append(line)
        finally:
            if openedFile:
                openedFile.close()
        return (resp, lines)

    def _shortcmd(self, line):
        """Internal: send a command and get the response.
        Same return value as _getresp()."""
        self._putcmd(line)
        return self._getresp()

    def _longcmd(self, line, file=None):
        """Internal: send a command and get the response plus following text.
        Same return value as _getlongresp()."""
        self._putcmd(line)
        return self._getlongresp(file)

    def _longcmdstring(self, line, file=None):
        """Internal: send a command and get the response plus following text.
        Same as _longcmd() and _getlongresp(), except that the returned `lines`
        are unicode strings rather than bytes objects.
        """
        self._putcmd(line)
        resp, list = self._getlongresp(file)
        return (resp, [line.decode(self.encoding, self.errors) for line in list])

    def _getoverviewfmt(self):
        """Internal: get the overview format. Queries the server if not
        already done, else returns the cached value."""
        try:
            return self._cachedoverviewfmt
        except AttributeError:
            pass
        try:
            resp, lines = self._longcmdstring('LIST OVERVIEW.FMT')
        except NNTPPermanentError:
            fmt = _DEFAULT_OVERVIEW_FMT[:]
        else:
            fmt = _parse_overview_fmt(lines)
        self._cachedoverviewfmt = fmt
        return fmt

    def _grouplist(self, lines):
        return [GroupInfo(*line.split()) for line in lines]

    def capabilities(self):
        """Process a CAPABILITIES command.  Not supported by all servers.
        Return:
        - resp: server response if successful
        - caps: a dictionary mapping capability names to lists of tokens
        (for example {'VERSION': ['2'], 'OVER': [], LIST: ['ACTIVE', 'HEADERS'] })
        """
        caps = {}
        resp, lines = self._longcmdstring('CAPABILITIES')
        for line in lines:
            name, *tokens = line.split()
            caps[name] = tokens
        return (resp, caps)

    def newgroups(self, date, *, file=None):
        """Process a NEWGROUPS command.  Arguments:
        - date: a date or datetime object
        Return:
        - resp: server response if successful
        - list: list of newsgroup names
        """
        if not isinstance(date, (datetime.date, datetime.date)):
            raise TypeError("the date parameter must be a date or datetime object, not '{:40}'".format(date.__class__.__name__))
        date_str, time_str = _unparse_datetime(date, self.nntp_version < 2)
        cmd = 'NEWGROUPS {0} {1}'.format(date_str, time_str)
        resp, lines = self._longcmdstring(cmd, file)
        return (resp, self._grouplist(lines))

    def newnews(self, group, date, *, file=None):
        """Process a NEWNEWS command.  Arguments:
        - group: group name or '*'
        - date: a date or datetime object
        Return:
        - resp: server response if successful
        - list: list of message ids
        """
        if not isinstance(date, (datetime.date, datetime.date)):
            raise TypeError("the date parameter must be a date or datetime object, not '{:40}'".format(date.__class__.__name__))
        date_str, time_str = _unparse_datetime(date, self.nntp_version < 2)
        cmd = 'NEWNEWS {0} {1} {2}'.format(group, date_str, time_str)
        return self._longcmdstring(cmd, file)

    def list(self, group_pattern=None, *, file=None):
        """Process a LIST or LIST ACTIVE command. Arguments:
        - group_pattern: a pattern indicating which groups to query
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of (group, last, first, flag) (strings)
        """
        if group_pattern is not None:
            command = 'LIST ACTIVE ' + group_pattern
        else:
            command = 'LIST'
        resp, lines = self._longcmdstring(command, file)
        return (resp, self._grouplist(lines))

    def _getdescriptions(self, group_pattern, return_all):
        line_pat = re.compile('^(?P<group>[^ \t]+)[ \t]+(.*)$')
        resp, lines = self._longcmdstring('LIST NEWSGROUPS ' + group_pattern)
        if not resp.startswith('215'):
            resp, lines = self._longcmdstring('XGTITLE ' + group_pattern)
        groups = {}
        for raw_line in lines:
            match = line_pat.search(raw_line.strip())
            if match:
                name, desc = match.group(1, 2)
                if not return_all:
                    return desc
                groups[name] = desc
        if return_all:
            return (resp, groups)
        else:
            return ''

    def description(self, group):
        """Get a description for a single group.  If more than one
        group matches ('group' is a pattern), return the first.  If no
        group matches, return an empty string.

        This elides the response code from the server, since it can
        only be '215' or '285' (for xgtitle) anyway.  If the response
        code is needed, use the 'descriptions' method.

        NOTE: This neither checks for a wildcard in 'group' nor does
        it check whether the group actually exists."""
        return self._getdescriptions(group, False)

    def descriptions(self, group_pattern):
        """Get descriptions for a range of groups."""
        return self._getdescriptions(group_pattern, True)

    def group(self, name):
        """Process a GROUP command.  Argument:
        - group: the group name
        Returns:
        - resp: server response if successful
        - count: number of articles
        - first: first article number
        - last: last article number
        - name: the group name
        """
        resp = self._shortcmd('GROUP ' + name)
        if not resp.startswith('211'):
            raise NNTPReplyError(resp)
        words = resp.split()
        count = first = last = 0
        n = len(words)
        if n > 1:
            count = words[1]
            if n > 2:
                first = words[2]
                if n > 3:
                    last = words[3]
                    if n > 4:
                        name = words[4].lower()
        return (resp, int(count), int(first), int(last), name)

    def help(self, *, file=None):
        """Process a HELP command. Argument:
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of strings returned by the server in response to the
                HELP command
        """
        return self._longcmdstring('HELP', file)

    def _statparse(self, resp):
        """Internal: parse the response line of a STAT, NEXT, LAST,
        ARTICLE, HEAD or BODY command."""
        if not resp.startswith('22'):
            raise NNTPReplyError(resp)
        words = resp.split()
        art_num = int(words[1])
        message_id = words[2]
        return (resp, art_num, message_id)

    def _statcmd(self, line):
        """Internal: process a STAT, NEXT or LAST command."""
        resp = self._shortcmd(line)
        return self._statparse(resp)

    def stat(self, message_spec=None):
        """Process a STAT command.  Argument:
        - message_spec: article number or message id (if not specified,
          the current article is selected)
        Returns:
        - resp: server response if successful
        - art_num: the article number
        - message_id: the message id
        """
        if message_spec:
            return self._statcmd('STAT {0}'.format(message_spec))
        else:
            return self._statcmd('STAT')

    def next(self):
        """Process a NEXT command.  No arguments.  Return as for STAT."""
        return self._statcmd('NEXT')

    def last(self):
        """Process a LAST command.  No arguments.  Return as for STAT."""
        return self._statcmd('LAST')

    def _artcmd(self, line, file=None):
        """Internal: process a HEAD, BODY or ARTICLE command."""
        resp, lines = self._longcmd(line, file)
        resp, art_num, message_id = self._statparse(resp)
        return (resp, ArticleInfo(art_num, message_id, lines))

    def head(self, message_spec=None, *, file=None):
        """Process a HEAD command.  Argument:
        - message_spec: article number or message id
        - file: filename string or file object to store the headers in
        Returns:
        - resp: server response if successful
        - ArticleInfo: (article number, message id, list of header lines)
        """
        if message_spec is not None:
            cmd = 'HEAD {0}'.format(message_spec)
        else:
            cmd = 'HEAD'
        return self._artcmd(cmd, file)

    def body(self, message_spec=None, *, file=None):
        """Process a BODY command.  Argument:
        - message_spec: article number or message id
        - file: filename string or file object to store the body in
        Returns:
        - resp: server response if successful
        - ArticleInfo: (article number, message id, list of body lines)
        """
        if message_spec is not None:
            cmd = 'BODY {0}'.format(message_spec)
        else:
            cmd = 'BODY'
        return self._artcmd(cmd, file)

    def article(self, message_spec=None, *, file=None):
        """Process an ARTICLE command.  Argument:
        - message_spec: article number or message id
        - file: filename string or file object to store the article in
        Returns:
        - resp: server response if successful
        - ArticleInfo: (article number, message id, list of article lines)
        """
        if message_spec is not None:
            cmd = 'ARTICLE {0}'.format(message_spec)
        else:
            cmd = 'ARTICLE'
        return self._artcmd(cmd, file)

    def slave(self):
        """Process a SLAVE command.  Returns:
        - resp: server response if successful
        """
        return self._shortcmd('SLAVE')

    def xhdr(self, hdr, str, *, file=None):
        """Process an XHDR command (optional server extension).  Arguments:
        - hdr: the header type (e.g. 'subject')
        - str: an article nr, a message id, or a range nr1-nr2
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of (nr, value) strings
        """
        pat = re.compile('^([0-9]+) ?(.*)\n?')
        resp, lines = self._longcmdstring('XHDR {0} {1}'.format(hdr, str), file)

        def remove_number(line):
            m = pat.match(line)
            return m.group(1, 2) if m else line
        return (resp, [remove_number(line) for line in lines])

    def xover(self, start, end, *, file=None):
        """Process an XOVER command (optional server extension) Arguments:
        - start: start of range
        - end: end of range
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of dicts containing the response fields
        """
        resp, lines = self._longcmdstring('XOVER {0}-{1}'.format(start, end), file)
        fmt = self._getoverviewfmt()
        return (resp, _parse_overview(lines, fmt))

    def over(self, message_spec, *, file=None):
        """Process an OVER command.  If the command isn't supported, fall
        back to XOVER. Arguments:
        - message_spec:
            - either a message id, indicating the article to fetch
              information about
            - or a (start, end) tuple, indicating a range of article numbers;
              if end is None, information up to the newest message will be
              retrieved
            - or None, indicating the current article number must be used
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of dicts containing the response fields

        NOTE: the "message id" form isn't supported by XOVER
        """
        cmd = 'OVER' if 'OVER' in self._caps else 'XOVER'
        if isinstance(message_spec, (tuple, list)):
            start, end = message_spec
            cmd += ' {0}-{1}'.format(start, end or '')
        elif message_spec is not None:
            cmd = cmd + ' ' + message_spec
        resp, lines = self._longcmdstring(cmd, file)
        fmt = self._getoverviewfmt()
        return (resp, _parse_overview(lines, fmt))

    def date(self):
        """Process the DATE command.
        Returns:
        - resp: server response if successful
        - date: datetime object
        """
        resp = self._shortcmd('DATE')
        if not resp.startswith('111'):
            raise NNTPReplyError(resp)
        elem = resp.split()
        if len(elem) != 2:
            raise NNTPDataError(resp)
        date = elem[1]
        if len(date) != 14:
            raise NNTPDataError(resp)
        return (resp, _parse_datetime(date, None))

    def _post(self, command, f):
        resp = self._shortcmd(command)
        if not resp.startswith('3'):
            raise NNTPReplyError(resp)
        if isinstance(f, (bytes, bytearray)):
            f = f.splitlines()
        for line in f:
            if not line.endswith(_CRLF):
                line = line.rstrip(b'\r\n') + _CRLF
            if line.startswith(b'.'):
                line = b'.' + line
            self.file.write(line)
        self.file.write(b'.\r\n')
        self.file.flush()
        return self._getresp()

    def post(self, data):
        """Process a POST command.  Arguments:
        - data: bytes object, iterable or file containing the article
        Returns:
        - resp: server response if successful"""
        return self._post('POST', data)

    def ihave(self, message_id, data):
        """Process an IHAVE command.  Arguments:
        - message_id: message-id of the article
        - data: file containing the article
        Returns:
        - resp: server response if successful
        Note that if the server refuses the article an exception is raised."""
        return self._post('IHAVE {0}'.format(message_id), data)

    def _close(self):
        try:
            if self.file:
                self.file.close()
                del self.file
        finally:
            self.sock.close()

    def quit(self):
        """Process a QUIT command and close the socket.  Returns:
        - resp: server response if successful"""
        try:
            resp = self._shortcmd('QUIT')
        finally:
            self._close()
        return resp

    def login(self, user=None, password=None, usenetrc=True):
        if self.authenticated:
            raise ValueError('Already logged in.')
        if not user and (not usenetrc):
            raise ValueError('At least one of `user` and `usenetrc` must be specified')
        try:
            if usenetrc and (not user):
                import netrc
                credentials = netrc.netrc()
                auth = credentials.authenticators(self.host)
                if auth:
                    user = auth[0]
                    password = auth[2]
        except OSError:
            pass
        if not user:
            return
        resp = self._shortcmd('authinfo user ' + user)
        if resp.startswith('381'):
            if not password:
                raise NNTPReplyError(resp)
            else:
                resp = self._shortcmd('authinfo pass ' + password)
                if not resp.startswith('281'):
                    raise NNTPPermanentError(resp)
        self._caps = None
        self.getcapabilities()
        if self.readermode_afterauth and 'READER' not in self._caps:
            self._setreadermode()
            self._caps = None
            self.getcapabilities()

    def _setreadermode(self):
        try:
            self.welcome = self._shortcmd('mode reader')
        except NNTPPermanentError:
            pass
        except NNTPTemporaryError as e:
            if e.response.startswith('480'):
                self.readermode_afterauth = True
            else:
                raise
    if _have_ssl:

        def starttls(self, context=None):
            """Process a STARTTLS command. Arguments:
            - context: SSL context to use for the encrypted connection
            """
            if self.tls_on:
                raise ValueError('TLS is already enabled.')
            if self.authenticated:
                raise ValueError('TLS cannot be started after authentication.')
            resp = self._shortcmd('STARTTLS')
            if resp.startswith('382'):
                self.file.close()
                self.sock = _encrypt_on(self.sock, context, self.host)
                self.file = self.sock.makefile('rwb')
                self.tls_on = True
                self._caps = None
                self.getcapabilities()
            else:
                raise NNTPError('TLS failed to start.')