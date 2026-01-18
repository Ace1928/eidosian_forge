import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _tunnel(self):
    connect = b'CONNECT %s:%d HTTP/1.0\r\n' % (self._tunnel_host.encode('ascii'), self._tunnel_port)
    headers = [connect]
    for header, value in self._tunnel_headers.items():
        headers.append(f'{header}: {value}\r\n'.encode('latin-1'))
    headers.append(b'\r\n')
    self.send(b''.join(headers))
    del headers
    response = self.response_class(self.sock, method=self._method)
    try:
        version, code, message = response._read_status()
        if code != http.HTTPStatus.OK:
            self.close()
            raise OSError(f'Tunnel connection failed: {code} {message.strip()}')
        while True:
            line = response.fp.readline(_MAXLINE + 1)
            if len(line) > _MAXLINE:
                raise LineTooLong('header line')
            if not line:
                break
            if line in (b'\r\n', b'\n', b''):
                break
            if self.debuglevel > 0:
                print('header:', line.decode())
    finally:
        response.close()