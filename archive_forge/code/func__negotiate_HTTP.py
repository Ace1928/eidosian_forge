from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def _negotiate_HTTP(self, dest_addr, dest_port):
    """Negotiates a connection through an HTTP server.

        NOTE: This currently only supports HTTP CONNECT-style proxies."""
    proxy_type, addr, port, rdns, username, password = self.proxy
    addr = dest_addr if rdns else socket.gethostbyname(dest_addr)
    http_headers = [b'CONNECT ' + addr.encode('idna') + b':' + str(dest_port).encode() + b' HTTP/1.1', b'Host: ' + dest_addr.encode('idna')]
    if username and password:
        http_headers.append(b'Proxy-Authorization: basic ' + b64encode(username + b':' + password))
    http_headers.append(b'\r\n')
    self.sendall(b'\r\n'.join(http_headers))
    fobj = self.makefile()
    status_line = fobj.readline()
    fobj.close()
    if not status_line:
        raise GeneralProxyError('Connection closed unexpectedly')
    try:
        proto, status_code, status_msg = status_line.split(' ', 2)
    except ValueError:
        raise GeneralProxyError('HTTP proxy server sent invalid response')
    if not proto.startswith('HTTP/'):
        raise GeneralProxyError('Proxy server does not appear to be an HTTP proxy')
    try:
        status_code = int(status_code)
    except ValueError:
        raise HTTPError('HTTP proxy server did not return a valid HTTP status')
    if status_code != 200:
        error = '{0}: {1}'.format(status_code, status_msg)
        if status_code in (400, 403, 405):
            error += '\n[*] Note: The HTTP proxy server may not be supported by PySocks (must be a CONNECT tunnel proxy)'
        raise HTTPError(error)
    self.proxy_sockname = (b'0.0.0.0', 0)
    self.proxy_peername = (addr, dest_port)