import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
@_PathAttribute.register_type(BGP_ATTR_TYPE_MP_REACH_NLRI)
class BGPPathAttributeMpReachNLRI(_PathAttribute):
    _VALUE_PACK_STR = '!HBB'
    _VALUE_PACK_SIZE = struct.calcsize(_VALUE_PACK_STR)
    _RD_LENGTH = 8
    _RESERVED_LENGTH = 1
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL
    _class_suffixes = ['AddrPrefix']
    _opt_attributes = ['next_hop']
    _TYPE = {'ascii': ['next_hop']}

    def __init__(self, afi, safi, next_hop, nlri, flags=0, type_=None, length=None):
        super(BGPPathAttributeMpReachNLRI, self).__init__(flags=flags, type_=type_, length=length)
        self.afi = afi
        self.safi = safi
        if not isinstance(next_hop, (list, tuple)):
            next_hop = [next_hop]
        for n in next_hop:
            if not ip.valid_ipv4(n) and (not ip.valid_ipv6(n)):
                raise ValueError('Invalid address for next_hop: %s' % n)
        if next_hop:
            self._next_hop = next_hop[0]
        else:
            self._next_hop = None
        self._next_hop_list = next_hop
        self.nlri = nlri
        addr_cls = _get_addr_class(afi, safi)
        for i in nlri:
            if not isinstance(i, addr_cls):
                raise ValueError('Invalid NRLI class for afi=%d and safi=%d' % (self.afi, self.safi))

    @staticmethod
    def split_bin_with_len(buf, unit_len):
        f = io.BytesIO(buf)
        return [f.read(unit_len) for _ in range(0, len(buf), unit_len)]

    @classmethod
    def parse_next_hop_ipv4(cls, buf, unit_len):
        next_hop = []
        for next_hop_bin in cls.split_bin_with_len(buf, unit_len):
            next_hop.append(addrconv.ipv4.bin_to_text(next_hop_bin[-4:]))
        return next_hop

    @classmethod
    def parse_next_hop_ipv6(cls, buf, unit_len):
        next_hop = []
        for next_hop_bin in cls.split_bin_with_len(buf, unit_len):
            next_hop.append(addrconv.ipv6.bin_to_text(next_hop_bin[-16:]))
        return next_hop

    @classmethod
    def parse_value(cls, buf):
        afi, safi, next_hop_len = struct.unpack_from(cls._VALUE_PACK_STR, bytes(buf))
        rest = buf[cls._VALUE_PACK_SIZE:]
        next_hop_bin = rest[:next_hop_len]
        rest = rest[next_hop_len:]
        reserved = rest[:cls._RESERVED_LENGTH]
        assert reserved == b'\x00'
        nlri_bin = rest[cls._RESERVED_LENGTH:]
        addr_cls = _get_addr_class(afi, safi)
        nlri = []
        while nlri_bin:
            n, nlri_bin = addr_cls.parser(nlri_bin)
            nlri.append(n)
        rf = RouteFamily(afi, safi)
        if rf == RF_IPv4_VPN:
            next_hop = cls.parse_next_hop_ipv4(next_hop_bin, cls._RD_LENGTH + 4)
            next_hop_len -= cls._RD_LENGTH * len(next_hop)
        elif rf == RF_IPv6_VPN:
            next_hop = cls.parse_next_hop_ipv6(next_hop_bin, cls._RD_LENGTH + 16)
            next_hop_len -= cls._RD_LENGTH * len(next_hop)
        elif afi == addr_family.IP or (rf == RF_L2_EVPN and next_hop_len < 16):
            next_hop = cls.parse_next_hop_ipv4(next_hop_bin, 4)
        elif afi == addr_family.IP6 or (rf == RF_L2_EVPN and next_hop_len >= 16):
            next_hop = cls.parse_next_hop_ipv6(next_hop_bin, 16)
        elif rf == RF_L2VPN_FLOWSPEC:
            next_hop = []
        else:
            raise ValueError('Invalid address family: afi=%d, safi=%d' % (afi, safi))
        return {'afi': afi, 'safi': safi, 'next_hop': next_hop, 'nlri': nlri}

    def serialize_next_hop(self):
        buf = bytearray()
        for next_hop in self.next_hop_list:
            if self.afi == addr_family.IP6:
                next_hop = str(netaddr.IPAddress(next_hop).ipv6())
            next_hop_bin = ip.text_to_bin(next_hop)
            if RouteFamily(self.afi, self.safi) in (RF_IPv4_VPN, RF_IPv6_VPN):
                next_hop_bin = b'\x00' * self._RD_LENGTH + next_hop_bin
            buf += next_hop_bin
        return buf

    def serialize_value(self):
        next_hop_bin = self.serialize_next_hop()
        next_hop_len = len(next_hop_bin)
        buf = bytearray()
        msg_pack_into(self._VALUE_PACK_STR, buf, 0, self.afi, self.safi, next_hop_len)
        buf += next_hop_bin
        buf += b'\x00'
        nlri_bin = bytearray()
        for n in self.nlri:
            nlri_bin += n.serialize()
        buf += nlri_bin
        return buf

    @property
    def next_hop(self):
        return self._next_hop

    @next_hop.setter
    def next_hop(self, addr):
        if not ip.valid_ipv4(addr) and (not ip.valid_ipv6(addr)):
            raise ValueError('Invalid address for next_hop: %s' % addr)
        self._next_hop = addr
        self.next_hop_list[0] = addr

    @property
    def next_hop_list(self):
        return self._next_hop_list

    @next_hop_list.setter
    def next_hop_list(self, addr_list):
        if not isinstance(addr_list, (list, tuple)):
            addr_list = [addr_list]
        for addr in addr_list:
            if not ip.valid_ipv4(addr) and (not ip.valid_ipv6(addr)):
                raise ValueError('Invalid address for next_hop: %s' % addr)
        self._next_hop = addr_list[0]
        self._next_hop_list = addr_list

    @property
    def route_family(self):
        return _rf_map[self.afi, self.safi]