import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
def _do_ssl_handshake(self) -> None:
    try:
        self._handshake_reading = False
        self._handshake_writing = False
        self.socket.do_handshake()
    except ssl.SSLError as err:
        if err.args[0] == ssl.SSL_ERROR_WANT_READ:
            self._handshake_reading = True
            return
        elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
            self._handshake_writing = True
            return
        elif err.args[0] in (ssl.SSL_ERROR_EOF, ssl.SSL_ERROR_ZERO_RETURN):
            return self.close(exc_info=err)
        elif err.args[0] == ssl.SSL_ERROR_SSL:
            try:
                peer = self.socket.getpeername()
            except Exception:
                peer = '(not connected)'
            gen_log.warning('SSL Error on %s %s: %s', self.socket.fileno(), peer, err)
            return self.close(exc_info=err)
        raise
    except ssl.CertificateError as err:
        return self.close(exc_info=err)
    except socket.error as err:
        if self._is_connreset(err) or err.args[0] in (0, errno.EBADF, errno.ENOTCONN):
            return self.close(exc_info=err)
        raise
    except AttributeError as err:
        return self.close(exc_info=err)
    else:
        self._ssl_accepting = False
        assert ssl.HAS_SNI
        self._finish_ssl_connect()