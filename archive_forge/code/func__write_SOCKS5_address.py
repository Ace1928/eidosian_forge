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
def _write_SOCKS5_address(self, addr, file):
    """
        Return the host and port packed for the SOCKS5 protocol,
        and the resolved address as a tuple object.
        """
    host, port = addr
    proxy_type, _, _, rdns, username, password = self.proxy
    family_to_byte = {socket.AF_INET: b'\x01', socket.AF_INET6: b'\x04'}
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            addr_bytes = socket.inet_pton(family, host)
            file.write(family_to_byte[family] + addr_bytes)
            host = socket.inet_ntop(family, addr_bytes)
            file.write(struct.pack('>H', port))
            return (host, port)
        except socket.error:
            continue
    if rdns:
        host_bytes = host.encode('idna')
        file.write(b'\x03' + chr(len(host_bytes)).encode() + host_bytes)
    else:
        addresses = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_ADDRCONFIG)
        target_addr = addresses[0]
        family = target_addr[0]
        host = target_addr[4][0]
        addr_bytes = socket.inet_pton(family, host)
        file.write(family_to_byte[family] + addr_bytes)
        host = socket.inet_ntop(family, addr_bytes)
    file.write(struct.pack('>H', port))
    return (host, port)