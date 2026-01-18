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
class _ZebraIPNexthopLookup(_ZebraMessageBody, metaclass=abc.ABCMeta):
    """
    Base class for ZEBRA_IPV4_NEXTHOP_LOOKUP and
    ZEBRA_IPV6_NEXTHOP_LOOKUP message body.
    """
    _METRIC_FMT = '!I'
    METRIC_SIZE = struct.calcsize(_METRIC_FMT)
    ADDR_CLS = None
    ADDR_LEN = None

    def __init__(self, addr, metric=None, nexthops=None):
        super(_ZebraIPNexthopLookup, self).__init__()
        assert ip.valid_ipv4(addr) or ip.valid_ipv6(addr)
        self.addr = addr
        self.metric = metric
        nexthops = nexthops or []
        for nexthop in nexthops:
            assert isinstance(nexthop, _NextHop)
        self.nexthops = nexthops

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        addr = cls.ADDR_CLS.bin_to_text(buf[:cls.ADDR_LEN])
        rest = buf[cls.ADDR_LEN:]
        metric = None
        if rest:
            metric, = struct.unpack_from(cls._METRIC_FMT, rest)
            rest = rest[cls.METRIC_SIZE:]
        nexthops = None
        if rest:
            nexthops, rest = _parse_nexthops(rest, version)
        return cls(addr, metric, nexthops)

    def serialize(self, version=_DEFAULT_VERSION):
        buf = self.ADDR_CLS.text_to_bin(self.addr)
        if self.metric is None:
            return buf
        buf += struct.pack(self._METRIC_FMT, self.metric)
        return buf + _serialize_nexthops(self.nexthops, version=version)