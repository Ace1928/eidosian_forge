import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
class _NextHop(type_desc.TypeDisp, stringify.StringifyMixin, metaclass=abc.ABCMeta):
    """
    Base class for Zebra Nexthop structure.
    """
    _HEADER_FMT = '!B'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, ifindex=None, ifname=None, addr=None, type_=None):
        super(_NextHop, self).__init__()
        self.ifindex = ifindex
        self.ifname = ifname
        self.addr = addr
        self.type = type_

    @classmethod
    @abc.abstractmethod
    def parse(cls, buf):
        type_, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        subcls = cls._lookup_type(type_)
        if subcls is None:
            raise struct.error('unsupported Nexthop type: %d' % type_)
        nexthop, rest = subcls.parse(rest)
        nexthop.type = type_
        return (nexthop, rest)

    @abc.abstractmethod
    def _serialize(self):
        return b''

    def serialize(self, version=_DEFAULT_VERSION):
        if self.type is None:
            if version <= 3:
                nh_cls = _NextHop
            elif version == 4:
                nh_cls = _FrrNextHop
            else:
                raise ValueError('Unsupported Zebra protocol version: %d' % version)
            self.type = nh_cls._rev_lookup_type(self.__class__)
        return struct.pack(self._HEADER_FMT, self.type) + self._serialize()