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
class _RouteDistinguisher(StringifyMixin, TypeDisp, _Value):
    _PACK_STR = '!H'
    TWO_OCTET_AS = 0
    IPV4_ADDRESS = 1
    FOUR_OCTET_AS = 2

    def __init__(self, admin=0, assigned=0, type_=None):
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_
        self.admin = admin
        self.assigned = assigned

    @classmethod
    def parser(cls, buf):
        assert len(buf) == 8
        type_, = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        subcls = cls._lookup_type(type_)
        return subcls(**subcls.parse_value(rest))

    @classmethod
    def from_str(cls, str_):
        assert isinstance(str_, str)
        first, second = str_.split(':')
        if '.' in first:
            type_ = cls.IPV4_ADDRESS
        elif int(first) > 1 << 16:
            type_ = cls.FOUR_OCTET_AS
            first = int(first)
        else:
            type_ = cls.TWO_OCTET_AS
            first = int(first)
        subcls = cls._lookup_type(type_)
        return subcls(admin=first, assigned=int(second))

    def serialize(self):
        value = self.serialize_value()
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.type)
        return bytes(buf + value)

    @property
    def formatted_str(self):
        return '%s:%s' % (self.admin, self.assigned)