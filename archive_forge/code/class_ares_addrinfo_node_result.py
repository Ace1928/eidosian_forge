from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
class ares_addrinfo_node_result(AresResult):
    __slots__ = ('ttl', 'flags', 'family', 'socktype', 'protocol', 'addr')

    def __init__(self, ares_node):
        self.ttl = ares_node.ai_ttl
        self.flags = ares_node.ai_flags
        self.socktype = ares_node.ai_socktype
        self.protocol = ares_node.ai_protocol
        addr = ares_node.ai_addr
        assert addr.sa_family == ares_node.ai_family
        ip = _ffi.new('char []', _lib.INET6_ADDRSTRLEN)
        if addr.sa_family == socket.AF_INET:
            self.family = socket.AF_INET
            s = _ffi.cast('struct sockaddr_in*', addr)
            if _ffi.NULL != _lib.ares_inet_ntop(s.sin_family, _ffi.addressof(s.sin_addr), ip, _lib.INET6_ADDRSTRLEN):
                self.addr = (_ffi.string(ip, _lib.INET6_ADDRSTRLEN), socket.ntohs(s.sin_port))
        elif addr.sa_family == socket.AF_INET6:
            self.family = socket.AF_INET6
            s = _ffi.cast('struct sockaddr_in6*', addr)
            if _ffi.NULL != _lib.ares_inet_ntop(s.sin6_family, _ffi.addressof(s.sin6_addr), ip, _lib.INET6_ADDRSTRLEN):
                self.addr = (_ffi.string(ip, _lib.INET6_ADDRSTRLEN), socket.ntohs(s.sin6_port), s.sin6_flowinfo, s.sin6_scope_id)
        else:
            raise ValueError('invalid sockaddr family')