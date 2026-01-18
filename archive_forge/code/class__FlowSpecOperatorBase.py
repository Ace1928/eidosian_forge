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
class _FlowSpecOperatorBase(_FlowSpecComponentBase):
    """Operator base class for Flow Specification NLRI component

    ===================== ===============================================
    Attribute             Description
    ===================== ===============================================
    operator              Match conditions.
    value                 Value of component.
    ===================== ===============================================
    """
    _OPE_PACK_STR = '!B'
    _OPE_PACK_STR_SIZE = struct.calcsize(_OPE_PACK_STR)
    _VAL_PACK_STR = '!%ds'
    END_OF_LIST = 1 << 7
    AND = 1 << 6
    OR = 0
    _LENGTH_BIT_MASK = 48
    _logical_conditions = {'|': OR, '&': AND}
    _comparison_conditions = {}

    def __init__(self, operator, value, type_=None):
        super(_FlowSpecOperatorBase, self).__init__(type_)
        self.operator = operator
        self.value = value

    @classmethod
    def parse_body(cls, rest):
        operator, = struct.unpack_from(cls._OPE_PACK_STR, bytes(rest))
        rest = rest[cls._OPE_PACK_STR_SIZE:]
        length = 1 << ((operator & cls._LENGTH_BIT_MASK) >> 4)
        value_type = type_desc.IntDescr(length)
        value = value_type.to_user(rest)
        rest = rest[length:]
        return (cls(operator, value), rest)

    def serialize_body(self):
        byte_length = (self.value.bit_length() + 7) // 8 or 1
        length = int(math.ceil(math.log(byte_length, 2)))
        self.operator |= length << 4
        buf = struct.pack(self._OPE_PACK_STR, self.operator)
        value_type = type_desc.IntDescr(1 << length)
        buf += struct.pack(self._VAL_PACK_STR % (1 << length), value_type.from_user(self.value))
        return buf

    @classmethod
    def from_str(cls, val):
        operator = 0
        rules = []
        elements = [v.strip() for v in re.split('([0-9]+)|([A-Z]+)|(\\|&\\+)|([!=<>]+)', val) if v and v.strip()]
        elms_iter = iter(elements)
        for elm in elms_iter:
            if elm in cls._logical_conditions:
                operator |= cls._logical_conditions[elm]
                continue
            elif elm in cls._comparison_conditions:
                operator |= cls._comparison_conditions[elm]
                continue
            elif elm == '+':
                rules[-1].value |= cls._to_value(next(elms_iter))
                continue
            value = cls._to_value(elm)
            operator = cls.normalize_operator(operator)
            rules.append(cls(operator, value))
            operator = 0
        return rules

    @classmethod
    def _to_value(cls, value):
        return value

    @classmethod
    def normalize_operator(cls, operator):
        return operator