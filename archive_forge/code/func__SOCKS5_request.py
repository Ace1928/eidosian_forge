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
def _SOCKS5_request(self, conn, cmd, dst):
    """
        Send SOCKS5 request with given command (CMD field) and
        address (DST field). Returns resolved DST address that was used.
        """
    proxy_type, addr, port, rdns, username, password = self.proxy
    writer = conn.makefile('wb')
    reader = conn.makefile('rb', 0)
    try:
        if username and password:
            writer.write(b'\x05\x02\x00\x02')
        else:
            writer.write(b'\x05\x01\x00')
        writer.flush()
        chosen_auth = self._readall(reader, 2)
        if chosen_auth[0:1] != b'\x05':
            raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
        if chosen_auth[1:2] == b'\x02':
            writer.write(b'\x01' + chr(len(username)).encode() + username + chr(len(password)).encode() + password)
            writer.flush()
            auth_status = self._readall(reader, 2)
            if auth_status[0:1] != b'\x01':
                raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
            if auth_status[1:2] != b'\x00':
                raise SOCKS5AuthError('SOCKS5 authentication failed')
        elif chosen_auth[1:2] != b'\x00':
            if chosen_auth[1:2] == b'\xff':
                raise SOCKS5AuthError('All offered SOCKS5 authentication methods were rejected')
            else:
                raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
        writer.write(b'\x05' + cmd + b'\x00')
        resolved = self._write_SOCKS5_address(dst, writer)
        writer.flush()
        resp = self._readall(reader, 3)
        if resp[0:1] != b'\x05':
            raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
        status = ord(resp[1:2])
        if status != 0:
            error = SOCKS5_ERRORS.get(status, 'Unknown error')
            raise SOCKS5Error('{0:#04x}: {1}'.format(status, error))
        bnd = self._read_SOCKS5_address(reader)
        super(socksocket, self).settimeout(self._timeout)
        return (resolved, bnd)
    finally:
        reader.close()
        writer.close()