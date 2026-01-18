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
def _negotiate_SOCKS5(self, *dest_addr):
    """Negotiates a stream connection through a SOCKS5 server."""
    CONNECT = b'\x01'
    self.proxy_peername, self.proxy_sockname = self._SOCKS5_request(self, CONNECT, dest_addr)