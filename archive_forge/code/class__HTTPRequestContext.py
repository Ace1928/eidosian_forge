import socket
import ssl
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
import typing
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable
class _HTTPRequestContext(object):

    def __init__(self, stream: iostream.IOStream, address: Tuple, protocol: Optional[str], trusted_downstream: Optional[List[str]]=None) -> None:
        self.address = address
        if stream.socket is not None:
            self.address_family = stream.socket.family
        else:
            self.address_family = None
        if self.address_family in (socket.AF_INET, socket.AF_INET6) and address is not None:
            self.remote_ip = address[0]
        else:
            self.remote_ip = '0.0.0.0'
        if protocol:
            self.protocol = protocol
        elif isinstance(stream, iostream.SSLIOStream):
            self.protocol = 'https'
        else:
            self.protocol = 'http'
        self._orig_remote_ip = self.remote_ip
        self._orig_protocol = self.protocol
        self.trusted_downstream = set(trusted_downstream or [])

    def __str__(self) -> str:
        if self.address_family in (socket.AF_INET, socket.AF_INET6):
            return self.remote_ip
        elif isinstance(self.address, bytes):
            return native_str(self.address)
        else:
            return str(self.address)

    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None:
        """Rewrite the ``remote_ip`` and ``protocol`` fields."""
        ip = headers.get('X-Forwarded-For', self.remote_ip)
        for ip in (cand.strip() for cand in reversed(ip.split(','))):
            if ip not in self.trusted_downstream:
                break
        ip = headers.get('X-Real-Ip', ip)
        if netutil.is_valid_ip(ip):
            self.remote_ip = ip
        proto_header = headers.get('X-Scheme', headers.get('X-Forwarded-Proto', self.protocol))
        if proto_header:
            proto_header = proto_header.split(',')[-1].strip()
        if proto_header in ('http', 'https'):
            self.protocol = proto_header

    def _unapply_xheaders(self) -> None:
        """Undo changes from `_apply_xheaders`.

        Xheaders are per-request so they should not leak to the next
        request on the same connection.
        """
        self.remote_ip = self._orig_remote_ip
        self.protocol = self._orig_protocol