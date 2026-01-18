import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
class FTP_TLS(FTP):
    """A FTP subclass which adds TLS support to FTP as described
        in RFC-4217.

        Connect as usual to port 21 implicitly securing the FTP control
        connection before authenticating.

        Securing the data connection requires user to explicitly ask
        for it by calling prot_p() method.

        Usage example:
        >>> from ftplib import FTP_TLS
        >>> ftps = FTP_TLS('ftp.python.org')
        >>> ftps.login()  # login anonymously previously securing control channel
        '230 Guest login ok, access restrictions apply.'
        >>> ftps.prot_p()  # switch to secure data connection
        '200 Protection level set to P'
        >>> ftps.retrlines('LIST')  # list directory content securely
        total 9
        drwxr-xr-x   8 root     wheel        1024 Jan  3  1994 .
        drwxr-xr-x   8 root     wheel        1024 Jan  3  1994 ..
        drwxr-xr-x   2 root     wheel        1024 Jan  3  1994 bin
        drwxr-xr-x   2 root     wheel        1024 Jan  3  1994 etc
        d-wxrwxr-x   2 ftp      wheel        1024 Sep  5 13:43 incoming
        drwxr-xr-x   2 root     wheel        1024 Nov 17  1993 lib
        drwxr-xr-x   6 1094     wheel        1024 Sep 13 19:07 pub
        drwxr-xr-x   3 root     wheel        1024 Jan  3  1994 usr
        -rw-r--r--   1 root     root          312 Aug  1  1994 welcome.msg
        '226 Transfer complete.'
        >>> ftps.quit()
        '221 Goodbye.'
        >>>
        """
    ssl_version = ssl.PROTOCOL_TLS_CLIENT

    def __init__(self, host='', user='', passwd='', acct='', keyfile=None, certfile=None, context=None, timeout=_GLOBAL_DEFAULT_TIMEOUT, source_address=None, *, encoding='utf-8'):
        if context is not None and keyfile is not None:
            raise ValueError('context and keyfile arguments are mutually exclusive')
        if context is not None and certfile is not None:
            raise ValueError('context and certfile arguments are mutually exclusive')
        if keyfile is not None or certfile is not None:
            import warnings
            warnings.warn('keyfile and certfile are deprecated, use a custom context instead', DeprecationWarning, 2)
        self.keyfile = keyfile
        self.certfile = certfile
        if context is None:
            context = ssl._create_stdlib_context(self.ssl_version, certfile=certfile, keyfile=keyfile)
        self.context = context
        self._prot_p = False
        super().__init__(host, user, passwd, acct, timeout, source_address, encoding=encoding)

    def login(self, user='', passwd='', acct='', secure=True):
        if secure and (not isinstance(self.sock, ssl.SSLSocket)):
            self.auth()
        return super().login(user, passwd, acct)

    def auth(self):
        """Set up secure control connection by using TLS/SSL."""
        if isinstance(self.sock, ssl.SSLSocket):
            raise ValueError('Already using TLS')
        if self.ssl_version >= ssl.PROTOCOL_TLS:
            resp = self.voidcmd('AUTH TLS')
        else:
            resp = self.voidcmd('AUTH SSL')
        self.sock = self.context.wrap_socket(self.sock, server_hostname=self.host)
        self.file = self.sock.makefile(mode='r', encoding=self.encoding)
        return resp

    def ccc(self):
        """Switch back to a clear-text control connection."""
        if not isinstance(self.sock, ssl.SSLSocket):
            raise ValueError('not using TLS')
        resp = self.voidcmd('CCC')
        self.sock = self.sock.unwrap()
        return resp

    def prot_p(self):
        """Set up secure data connection."""
        self.voidcmd('PBSZ 0')
        resp = self.voidcmd('PROT P')
        self._prot_p = True
        return resp

    def prot_c(self):
        """Set up clear text data connection."""
        resp = self.voidcmd('PROT C')
        self._prot_p = False
        return resp

    def ntransfercmd(self, cmd, rest=None):
        conn, size = super().ntransfercmd(cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn, server_hostname=self.host)
        return (conn, size)

    def abort(self):
        line = b'ABOR' + B_CRLF
        self.sock.sendall(line)
        resp = self.getmultiline()
        if resp[:3] not in {'426', '225', '226'}:
            raise error_proto(resp)
        return resp