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
class _ExtendedCommunity(StringifyMixin, TypeDisp, _Value):
    _PACK_STR = '!B7s'
    _PACK_STR_SIZE = struct.calcsize(_PACK_STR)
    _SUBTYPE_PACK_STR = '!B'
    IANA_AUTHORITY = 128
    TRANSITIVE = 64
    _TYPE_HIGH_MASK = ~TRANSITIVE
    TWO_OCTET_AS_SPECIFIC = 0
    IPV4_ADDRESS_SPECIFIC = 1
    FOUR_OCTET_AS_SPECIFIC = 2
    OPAQUE = 3
    SUBTYPE_ENCAPSULATION = 12
    ENCAPSULATION = (OPAQUE, SUBTYPE_ENCAPSULATION)
    EVPN = 6
    SUBTYPE_EVPN_MAC_MOBILITY = 0
    SUBTYPE_EVPN_ESI_LABEL = 1
    SUBTYPE_EVPN_ES_IMPORT_RT = 2
    EVPN_MAC_MOBILITY = (EVPN, SUBTYPE_EVPN_MAC_MOBILITY)
    EVPN_ESI_LABEL = (EVPN, SUBTYPE_EVPN_ESI_LABEL)
    EVPN_ES_IMPORT_RT = (EVPN, SUBTYPE_EVPN_ES_IMPORT_RT)
    FLOWSPEC = 128
    FLOWSPEC_L2VPN = 8
    SUBTYPE_FLOWSPEC_TRAFFIC_RATE = 6
    SUBTYPE_FLOWSPEC_TRAFFIC_ACTION = 7
    SUBTYPE_FLOWSPEC_REDIRECT = 8
    SUBTYPE_FLOWSPEC_TRAFFIC_REMARKING = 9
    SUBTYPE_FLOWSPEC_VLAN_ACTION = 10
    SUBTYPE_FLOWSPEC_TPID_ACTION = 11
    FLOWSPEC_TRAFFIC_RATE = (FLOWSPEC, SUBTYPE_FLOWSPEC_TRAFFIC_RATE)
    FLOWSPEC_TRAFFIC_ACTION = (FLOWSPEC, SUBTYPE_FLOWSPEC_TRAFFIC_ACTION)
    FLOWSPEC_REDIRECT = (FLOWSPEC, SUBTYPE_FLOWSPEC_REDIRECT)
    FLOWSPEC_TRAFFIC_REMARKING = (FLOWSPEC, SUBTYPE_FLOWSPEC_TRAFFIC_REMARKING)
    FLOWSPEC_VLAN_ACTION = (FLOWSPEC_L2VPN, SUBTYPE_FLOWSPEC_VLAN_ACTION)
    FLOWSPEC_TPID_ACTION = (FLOWSPEC_L2VPN, SUBTYPE_FLOWSPEC_TPID_ACTION)

    def __init__(self, type_=None):
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
            if isinstance(type_, (tuple, list)):
                type_ = type_[0]
        self.type = type_

    @classmethod
    def parse_subtype(cls, buf):
        subtype, = struct.unpack_from(cls._SUBTYPE_PACK_STR, buf)
        return subtype

    @classmethod
    def parse(cls, buf):
        type_, value = struct.unpack_from(cls._PACK_STR, buf)
        rest = buf[cls._PACK_STR_SIZE:]
        type_low = type_ & cls._TYPE_HIGH_MASK
        subtype = cls.parse_subtype(value)
        subcls = cls._lookup_type((type_low, subtype))
        if subcls == cls._UNKNOWN_TYPE:
            subcls = cls._lookup_type(type_low)
        return (subcls(type_=type_, **subcls.parse_value(value)), rest)

    def serialize(self):
        return struct.pack(self._PACK_STR, self.type, self.serialize_value())