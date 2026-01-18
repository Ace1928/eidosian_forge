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
@classmethod
def _parse_impl(cls, buf, version=_DEFAULT_VERSION, from_zebra=False):
    instance = None
    if version <= 3:
        route_type, flags, message = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
    elif version == 4:
        route_type, instance, flags, message = struct.unpack_from(cls._V4_HEADER_FMT, buf)
        rest = buf[cls.V4_HEADER_SIZE:]
    else:
        raise struct.error('Unsupported Zebra protocol version: %d' % version)
    if from_zebra:
        safi = None
    else:
        safi, = struct.unpack_from(cls._SAFI_FMT, rest)
        rest = rest[cls.SAFI_SIZE:]
    prefix, rest = _parse_ip_prefix(cls._FAMILY, rest)
    src_prefix = None
    if version == 4 and message & FRR_ZAPI_MESSAGE_SRCPFX:
        src_prefix, rest = _parse_ip_prefix(cls._FAMILY, rest)
    if from_zebra and message & ZAPI_MESSAGE_NEXTHOP:
        nexthops = []
        nexthop_num, = struct.unpack_from(cls._NUM_FMT, rest)
        rest = rest[cls.NUM_SIZE:]
        if cls._FAMILY == socket.AF_INET:
            for _ in range(nexthop_num):
                nexthop = addrconv.ipv4.bin_to_text(rest[:4])
                nexthops.append(nexthop)
                rest = rest[4:]
        else:
            for _ in range(nexthop_num):
                nexthop = addrconv.ipv6.bin_to_text(rest[:16])
                nexthops.append(nexthop)
                rest = rest[16:]
    else:
        nexthops, rest = _parse_nexthops(rest, version)
    ifindexes = []
    if from_zebra and message & ZAPI_MESSAGE_IFINDEX:
        ifindex_num, = struct.unpack_from(cls._NUM_FMT, rest)
        rest = rest[cls.NUM_SIZE:]
        for _ in range(ifindex_num):
            ifindex, = struct.unpack_from(cls._IFINDEX_FMT, rest)
            ifindexes.append(ifindex)
            rest = rest[cls.IFINDEX_SIZE:]
    if version <= 3:
        distance, rest = cls._parse_message_option(message, ZAPI_MESSAGE_DISTANCE, '!B', rest)
        metric, rest = cls._parse_message_option(message, ZAPI_MESSAGE_METRIC, '!I', rest)
        mtu, rest = cls._parse_message_option(message, ZAPI_MESSAGE_MTU, '!I', rest)
        tag, rest = cls._parse_message_option(message, ZAPI_MESSAGE_TAG, '!I', rest)
    elif version == 4:
        distance, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_DISTANCE, '!B', rest)
        metric, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_METRIC, '!I', rest)
        tag, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_TAG, '!I', rest)
        mtu, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_MTU, '!I', rest)
    else:
        raise struct.error('Unsupported Zebra protocol version: %d' % version)
    return cls(route_type, flags, message, safi, prefix, src_prefix, nexthops, ifindexes, distance, metric, mtu, tag, instance, from_zebra=from_zebra)