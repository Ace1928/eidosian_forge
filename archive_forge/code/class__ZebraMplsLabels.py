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
class _ZebraMplsLabels(_ZebraMessageBody):
    """
    Base class for ZEBRA_MPLS_LABELS_* message body.
    """
    _HEADER_FMT = '!B'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _FAMILY_FMT = '!I'
    FAMILY_SIZE = struct.calcsize(_FAMILY_FMT)
    _IPV4_PREFIX_FMT = '!4sB'
    _IPV6_PREFIX_FMT = '!16sB'
    IPV4_PREFIX_SIZE = struct.calcsize(_IPV4_PREFIX_FMT)
    IPV6_PREFIX_SIZE = struct.calcsize(_IPV6_PREFIX_FMT)
    _FAMILY_IPV4_PREFIX_FMT = '!I4sB'
    _FAMILY_IPV6_PREFIX_FMT = '!I16sB'
    _IFINDEX_FMT = '!I'
    IFINDEX_SIZE = struct.calcsize(_IFINDEX_FMT)
    _BODY_FMT = '!BII'

    def __init__(self, route_type, family, prefix, gate_addr, ifindex=None, distance=None, in_label=None, out_label=None):
        super(_ZebraMplsLabels, self).__init__()
        self.route_type = route_type
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix
        assert ip.valid_ipv4(gate_addr) or ip.valid_ipv6(gate_addr)
        self.gate_addr = gate_addr
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            assert ifindex is not None
        self.ifindex = ifindex
        assert distance is not None
        self.distance = distance
        assert in_label is not None
        self.in_label = in_label
        assert out_label is not None
        self.out_label = out_label

    @classmethod
    def _parse_family_prefix(cls, buf):
        family, = struct.unpack_from(cls._FAMILY_FMT, buf)
        rest = buf[cls.FAMILY_SIZE:]
        if socket.AF_INET == family:
            prefix, p_len = struct.unpack_from(cls._IPV4_PREFIX_FMT, rest)
            prefix = '%s/%d' % (addrconv.ipv4.bin_to_text(prefix), p_len)
            rest = rest[cls.IPV4_PREFIX_SIZE:]
        elif socket.AF_INET6 == family:
            prefix, p_len = struct.unpack_from(cls._IPV6_PREFIX_FMT, rest)
            prefix = '%s/%d' % (addrconv.ipv6.bin_to_text(prefix), p_len)
            rest = rest[cls.IPV6_PREFIX_SIZE:]
        else:
            raise struct.error('Unsupported family: %d' % family)
        return (family, prefix, rest)

    @classmethod
    def parse(cls, buf, version=_DEFAULT_FRR_VERSION):
        route_type, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        family, prefix, rest = cls._parse_family_prefix(rest)
        if family == socket.AF_INET:
            gate_addr = addrconv.ipv4.bin_to_text(rest[:4])
            rest = rest[4:]
        elif family == socket.AF_INET6:
            gate_addr = addrconv.ipv6.bin_to_text(rest[:16])
            rest = rest[16:]
        else:
            raise struct.error('Unsupported family: %d' % family)
        ifindex = None
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            ifindex, = struct.unpack_from(cls._IFINDEX_FMT, rest)
            rest = rest[cls.IFINDEX_SIZE:]
        distance, in_label, out_label = struct.unpack_from(cls._BODY_FMT, rest)
        return cls(route_type, family, prefix, gate_addr, ifindex, distance, in_label, out_label)

    def _serialize_family_prefix(self, prefix):
        if ip.valid_ipv4(prefix):
            family = socket.AF_INET
            prefix_addr, prefix_num = prefix.split('/')
            return (family, struct.pack(self._FAMILY_IPV4_PREFIX_FMT, family, addrconv.ipv4.text_to_bin(prefix_addr), int(prefix_num)))
        elif ip.valid_ipv6(prefix):
            family = socket.AF_INET6
            prefix_addr, prefix_num = prefix.split('/')
            return (family, struct.pack(self._FAMILY_IPV6_PREFIX_FMT, family, addrconv.ipv6.text_to_bin(prefix_addr), int(prefix_num)))
        raise ValueError('Invalid prefix: %s' % prefix)

    def serialize(self, version=_DEFAULT_FRR_VERSION):
        self.family, prefix_bin = self._serialize_family_prefix(self.prefix)
        if self.family == socket.AF_INET:
            gate_addr_bin = addrconv.ipv4.text_to_bin(self.gate_addr)
        elif self.family == socket.AF_INET6:
            gate_addr_bin = addrconv.ipv6.text_to_bin(self.gate_addr)
        else:
            raise ValueError('Unsupported family: %d' % self.family)
        body_bin = b''
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            body_bin = struct.pack(self._IFINDEX_FMT, self.ifindex)
        body_bin += struct.pack(self._BODY_FMT, self.distance, self.in_label, self.out_label)
        return struct.pack(self._HEADER_FMT, self.route_type) + prefix_bin + gate_addr_bin + body_bin