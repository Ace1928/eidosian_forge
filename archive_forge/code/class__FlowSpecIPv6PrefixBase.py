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
class _FlowSpecIPv6PrefixBase(_FlowSpecIPv6Component, IP6AddrPrefix):
    """
    Prefix base class for Flow Specification NLRI component
    """
    _PACK_STR = '!BB'

    def __init__(self, length, addr, offset=0, type_=None):
        super(_FlowSpecIPv6PrefixBase, self).__init__(type_)
        self.length = length
        self.offset = offset
        prefix = '%s/%s' % (addr, length)
        self.addr = str(netaddr.ip.IPNetwork(prefix).network)

    @classmethod
    def parser(cls, buf):
        length, offset = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        byte_length = (length + 7) // 8
        addr = cls._from_bin(rest[:byte_length])
        rest = rest[byte_length:]
        return (cls(length=length, offset=offset, addr=addr), rest)

    @classmethod
    def parse_body(cls, buf):
        return cls.parser(buf)

    def serialize(self):
        byte_length = (self.length + 7) // 8
        bin_addr = self._to_bin(self.addr)[:byte_length]
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.length, self.offset)
        return buf + bin_addr

    def serialize_body(self):
        return self.serialize()

    @classmethod
    def from_str(cls, value):
        rule = []
        values = value.split('/')
        if len(values) == 3:
            rule.append(cls(int(values[1]), values[0], offset=int(values[2])))
        else:
            rule.append(cls(int(values[1]), values[0]))
        return rule

    @property
    def value(self):
        return '%s/%s/%s' % (self.addr, self.length, self.offset)

    def to_str(self):
        return self.value