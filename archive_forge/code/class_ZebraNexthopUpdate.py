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
@_FrrZebraMessageBody.register_type(FRR_ZEBRA_NEXTHOP_UPDATE)
@_ZebraMessageBody.register_type(ZEBRA_NEXTHOP_UPDATE)
class ZebraNexthopUpdate(_ZebraMessageBody):
    """
    Message body class for ZEBRA_NEXTHOP_UPDATE.
    """
    _FAMILY_FMT = '!H'
    FAMILY_SIZE = struct.calcsize(_FAMILY_FMT)
    _DISTANCE_FMT = '!B'
    DISTANCE_SIZE = struct.calcsize(_DISTANCE_FMT)
    _METRIC_FMT = '!I'
    METRIC_SIZE = struct.calcsize(_METRIC_FMT)

    def __init__(self, family, prefix, distance=None, metric=None, nexthops=None):
        super(ZebraNexthopUpdate, self).__init__()
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            assert distance is not None
        self.distance = distance
        assert metric is not None
        self.metric = metric
        nexthops = nexthops or []
        for nexthop in nexthops:
            assert isinstance(nexthop, _NextHop)
        self.nexthops = nexthops

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        family, = struct.unpack_from(cls._FAMILY_FMT, buf)
        rest = buf[cls.FAMILY_SIZE:]
        prefix, rest = _parse_ip_prefix(family, rest)
        distance = None
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            distance, = struct.unpack_from(cls._DISTANCE_FMT, rest)
            rest = rest[cls.DISTANCE_SIZE:]
        metric, = struct.unpack_from(cls._METRIC_FMT, rest)
        rest = rest[cls.METRIC_SIZE:]
        nexthops, rest = _parse_nexthops(rest, version)
        return cls(family, prefix, distance, metric, nexthops)

    def serialize(self, version=_DEFAULT_VERSION):
        if ip.valid_ipv4(self.prefix):
            self.family = socket.AF_INET
        elif ip.valid_ipv6(self.prefix):
            self.family = socket.AF_INET6
        else:
            raise ValueError('Invalid prefix: %s' % self.prefix)
        buf = struct.pack(self._FAMILY_FMT, self.family)
        buf += _serialize_ip_prefix(self.prefix)
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            buf += struct.pack(self._DISTANCE_FMT, self.distance)
        buf += struct.pack(self._METRIC_FMT, self.metric)
        return buf + _serialize_nexthops(self.nexthops, version=version)