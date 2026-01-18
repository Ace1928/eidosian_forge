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
class _OptParam(StringifyMixin, TypeDisp, _Value):
    _PACK_STR = '!BB'

    def __init__(self, type_, value=None, length=None):
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_
        self.length = length
        if value is not None:
            self.value = value

    @classmethod
    def parser(cls, buf):
        type_, length = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        value = bytes(rest[:length])
        rest = rest[length:]
        subcls = cls._lookup_type(type_)
        caps = subcls.parse_value(value)
        if not isinstance(caps, list):
            caps = [subcls(type_=type_, length=length, **caps[0])]
        return (caps, rest)

    def serialize(self):
        value = self.serialize_value()
        self.length = len(value)
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.type, self.length)
        return buf + value