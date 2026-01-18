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