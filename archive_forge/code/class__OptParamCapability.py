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
@_OptParam.register_type(BGP_OPT_CAPABILITY)
class _OptParamCapability(_OptParam, TypeDisp):
    _CAP_HDR_PACK_STR = '!BB'

    def __init__(self, cap_code=None, cap_value=None, cap_length=None, type_=None, length=None):
        super(_OptParamCapability, self).__init__(type_=BGP_OPT_CAPABILITY, length=length)
        if cap_code is None:
            cap_code = self._rev_lookup_type(self.__class__)
        self.cap_code = cap_code
        if cap_value is not None:
            self.cap_value = cap_value
        if cap_length is not None:
            self.cap_length = cap_length

    @classmethod
    def parse_value(cls, buf):
        caps = []
        while len(buf) > 0:
            code, length = struct.unpack_from(cls._CAP_HDR_PACK_STR, bytes(buf))
            value = buf[struct.calcsize(cls._CAP_HDR_PACK_STR):]
            buf = buf[length + 2:]
            kwargs = {'cap_code': code, 'cap_length': length}
            subcls = cls._lookup_type(code)
            kwargs.update(subcls.parse_cap_value(value))
            caps.append(subcls(type_=BGP_OPT_CAPABILITY, length=length + 2, **kwargs))
        return caps

    def serialize_value(self):
        cap_value = self.serialize_cap_value()
        self.cap_length = len(cap_value)
        buf = bytearray()
        msg_pack_into(self._CAP_HDR_PACK_STR, buf, 0, self.cap_code, self.cap_length)
        return buf + cap_value