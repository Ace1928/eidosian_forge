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
class _FlowSpecL2VPNPrefixBase(_FlowSpecL2VPNComponent):
    """
    Prefix base class for Flow Specification NLRI component
    """
    _PACK_STR = '!B6s'

    def __init__(self, length, addr, type_=None):
        super(_FlowSpecL2VPNPrefixBase, self).__init__(type_)
        self.length = length
        self.addr = addr.lower()

    @classmethod
    def parse_body(cls, buf):
        length, addr = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        addr = addrconv.mac.bin_to_text(addr)
        return (cls(length=length, addr=addr), rest)

    def serialize(self):
        addr = addrconv.mac.text_to_bin(self.addr)
        return struct.pack(self._PACK_STR, self.length, addr)

    def serialize_body(self):
        return self.serialize()

    @classmethod
    def from_str(cls, value):
        return [cls(len(value.split(':')), value)]

    @property
    def value(self):
        return self.addr

    def to_str(self):
        return self.value