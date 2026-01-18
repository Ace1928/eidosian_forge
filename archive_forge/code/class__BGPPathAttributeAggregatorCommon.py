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
class _BGPPathAttributeAggregatorCommon(_PathAttribute):
    _VALUE_PACK_STR = None
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL | BGP_ATTR_FLAG_TRANSITIVE
    _TYPE = {'ascii': ['addr']}

    def __init__(self, as_number, addr, flags=0, type_=None, length=None):
        super(_BGPPathAttributeAggregatorCommon, self).__init__(flags=flags, type_=type_, length=length)
        self.as_number = as_number
        self.addr = addr

    @classmethod
    def parse_value(cls, buf):
        as_number, addr = struct.unpack_from(cls._VALUE_PACK_STR, bytes(buf))
        return {'as_number': as_number, 'addr': addrconv.ipv4.bin_to_text(addr)}

    def serialize_value(self):
        buf = bytearray()
        msg_pack_into(self._VALUE_PACK_STR, buf, 0, self.as_number, addrconv.ipv4.text_to_bin(self.addr))
        return buf