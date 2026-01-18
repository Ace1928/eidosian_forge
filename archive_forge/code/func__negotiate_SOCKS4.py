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
def _negotiate_SOCKS4(self, dest_addr, dest_port):
    """Negotiates a connection through a SOCKS4 server."""
    proxy_type, addr, port, rdns, username, password = self.proxy
    writer = self.makefile('wb')
    reader = self.makefile('rb', 0)
    try:
        remote_resolve = False
        try:
            addr_bytes = socket.inet_aton(dest_addr)
        except socket.error:
            if rdns:
                addr_bytes = b'\x00\x00\x00\x01'
                remote_resolve = True
            else:
                addr_bytes = socket.inet_aton(socket.gethostbyname(dest_addr))
        writer.write(struct.pack('>BBH', 4, 1, dest_port))
        writer.write(addr_bytes)
        if username:
            writer.write(username)
        writer.write(b'\x00')
        if remote_resolve:
            writer.write(dest_addr.encode('idna') + b'\x00')
        writer.flush()
        resp = self._readall(reader, 8)
        if resp[0:1] != b'\x00':
            raise GeneralProxyError('SOCKS4 proxy server sent invalid data')
        status = ord(resp[1:2])
        if status != 90:
            error = SOCKS4_ERRORS.get(status, 'Unknown error')
            raise SOCKS4Error('{0:#04x}: {1}'.format(status, error))
        self.proxy_sockname = (socket.inet_ntoa(resp[4:]), struct.unpack('>H', resp[2:4])[0])
        if remote_resolve:
            self.proxy_peername = (socket.inet_ntoa(addr_bytes), dest_port)
        else:
            self.proxy_peername = (dest_addr, dest_port)
    finally:
        reader.close()
        writer.close()