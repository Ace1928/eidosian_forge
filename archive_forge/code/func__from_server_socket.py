import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
def _from_server_socket(self, server_socket):
    try:
        s, addr = server_socket.accept()
        if self.server.stats['Enabled']:
            self.server.stats['Accepts'] += 1
        prevent_socket_inheritance(s)
        if hasattr(s, 'settimeout'):
            s.settimeout(self.server.timeout)
        mf = MakeFile
        ssl_env = {}
        if self.server.ssl_adapter is not None:
            try:
                s, ssl_env = self.server.ssl_adapter.wrap(s)
            except errors.NoSSLError:
                msg = 'The client sent a plain HTTP request, but this server only speaks HTTPS on this port.'
                buf = ['%s 400 Bad Request\r\n' % self.server.protocol, 'Content-Length: %s\r\n' % len(msg), 'Content-Type: text/plain\r\n\r\n', msg]
                wfile = mf(s, 'wb', io.DEFAULT_BUFFER_SIZE)
                try:
                    wfile.write(''.join(buf).encode('ISO-8859-1'))
                except OSError as ex:
                    if ex.args[0] not in errors.socket_errors_to_ignore:
                        raise
                return
            if not s:
                return
            mf = self.server.ssl_adapter.makefile
            if hasattr(s, 'settimeout'):
                s.settimeout(self.server.timeout)
        conn = self.server.ConnectionClass(self.server, s, mf)
        if not isinstance(self.server.bind_addr, (str, bytes)):
            if addr is None:
                if len(s.getsockname()) == 2:
                    addr = ('0.0.0.0', 0)
                else:
                    addr = ('::', 0)
            conn.remote_addr = addr[0]
            conn.remote_port = addr[1]
        conn.ssl_env = ssl_env
        return conn
    except socket.timeout:
        return
    except OSError as ex:
        if self.server.stats['Enabled']:
            self.server.stats['Socket Errors'] += 1
        if ex.args[0] in errors.socket_error_eintr:
            return
        if ex.args[0] in errors.socket_errors_nonblocking:
            return
        if ex.args[0] in errors.socket_errors_to_ignore:
            return
        raise