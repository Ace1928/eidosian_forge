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
@_PathAttribute.register_type(BGP_ATTR_TYPE_MP_UNREACH_NLRI)
class BGPPathAttributeMpUnreachNLRI(_PathAttribute):
    _VALUE_PACK_STR = '!HB'
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL
    _class_suffixes = ['AddrPrefix']

    def __init__(self, afi, safi, withdrawn_routes, flags=0, type_=None, length=None):
        super(BGPPathAttributeMpUnreachNLRI, self).__init__(flags=flags, type_=type_, length=length)
        self.afi = afi
        self.safi = safi
        self.withdrawn_routes = withdrawn_routes
        addr_cls = _get_addr_class(afi, safi)
        for i in withdrawn_routes:
            if not isinstance(i, addr_cls):
                raise ValueError('Invalid NRLI class for afi=%d and safi=%d' % (self.afi, self.safi))

    @classmethod
    def parse_value(cls, buf):
        afi, safi = struct.unpack_from(cls._VALUE_PACK_STR, bytes(buf))
        nlri_bin = buf[struct.calcsize(cls._VALUE_PACK_STR):]
        addr_cls = _get_addr_class(afi, safi)
        nlri = []
        while nlri_bin:
            n, nlri_bin = addr_cls.parser(nlri_bin)
            nlri.append(n)
        return {'afi': afi, 'safi': safi, 'withdrawn_routes': nlri}

    def serialize_value(self):
        buf = bytearray()
        msg_pack_into(self._VALUE_PACK_STR, buf, 0, self.afi, self.safi)
        nlri_bin = bytearray()
        for n in self.withdrawn_routes:
            nlri_bin += n.serialize()
        buf += nlri_bin
        return buf

    @property
    def route_family(self):
        return _rf_map[self.afi, self.safi]