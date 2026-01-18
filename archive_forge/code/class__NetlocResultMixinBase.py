from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class _NetlocResultMixinBase(object):
    """Shared methods for the parsed result objects containing a netloc element"""
    __slots__ = ()

    @property
    def username(self):
        return self._userinfo[0]

    @property
    def password(self):
        return self._userinfo[1]

    @property
    def hostname(self):
        hostname = self._hostinfo[0]
        if not hostname:
            return None
        separator = '%' if isinstance(hostname, str) else b'%'
        hostname, percent, zone = hostname.partition(separator)
        return hostname.lower() + percent + zone

    @property
    def port(self):
        port = self._hostinfo[1]
        if port is not None:
            if port.isdigit() and port.isascii():
                port = int(port)
            else:
                raise ValueError(f'Port could not be cast to integer value as {port!r}')
            if not 0 <= port <= 65535:
                raise ValueError('Port out of range 0-65535')
        return port
    __class_getitem__ = classmethod(types.GenericAlias)