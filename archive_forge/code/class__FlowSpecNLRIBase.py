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
class _FlowSpecNLRIBase(StringifyMixin, TypeDisp):
    """
    Base class for Flow Specification NLRI
    """
    ROUTE_FAMILY = None
    _LENGTH_SHORT_FMT = '!B'
    LENGTH_SHORT_SIZE = struct.calcsize(_LENGTH_SHORT_FMT)
    _LENGTH_LONG_FMT = '!H'
    LENGTH_LONG_SIZE = struct.calcsize(_LENGTH_LONG_FMT)
    _LENGTH_THRESHOLD = 61440
    FLOWSPEC_FAMILY = ''

    def __init__(self, length=0, rules=None):
        self.length = length
        rules = rules or []
        for r in rules:
            assert isinstance(r, _FlowSpecComponentBase)
        self.rules = rules

    @classmethod
    def parser(cls, buf):
        length, = struct.unpack_from(cls._LENGTH_LONG_FMT, bytes(buf))
        if length < cls._LENGTH_THRESHOLD:
            length >>= 8
            offset = cls.LENGTH_SHORT_SIZE
        else:
            offset = cls.LENGTH_LONG_SIZE
        kwargs = {'length': length}
        rest = buf[offset:offset + length]
        if cls.ROUTE_FAMILY.safi == subaddr_family.VPN_FLOWSPEC:
            route_dist = _RouteDistinguisher.parser(rest[:8])
            kwargs['route_dist'] = route_dist.formatted_str
            rest = rest[8:]
        rules = []
        while rest:
            subcls, rest = _FlowSpecComponentBase.parse_header(rest, cls.ROUTE_FAMILY.afi)
            while rest:
                rule, rest = subcls.parse_body(rest)
                rules.append(rule)
                if not isinstance(rule, _FlowSpecOperatorBase) or rule.operator & rule.END_OF_LIST:
                    break
        kwargs['rules'] = rules
        return (cls(**kwargs), rest)

    def serialize(self):
        rules_bin = b''
        if self.ROUTE_FAMILY.safi == subaddr_family.VPN_FLOWSPEC:
            route_dist = _RouteDistinguisher.from_str(self.route_dist)
            rules_bin += route_dist.serialize()
        self.rules.sort(key=lambda x: x.type)
        for _, rules in itertools.groupby(self.rules, key=lambda x: x.type):
            rules = list(rules)
            rules_bin += rules[0].serialize_header()
            if isinstance(rules[-1], _FlowSpecOperatorBase):
                rules[-1].operator |= rules[-1].END_OF_LIST
            for r in rules:
                rules_bin += r.serialize_body()
        self.length = len(rules_bin)
        if self.length < self._LENGTH_THRESHOLD:
            buf = struct.pack(self._LENGTH_SHORT_FMT, self.length)
        else:
            buf = struct.pack(self._LENGTH_LONG_FMT, self.length)
        return buf + rules_bin

    @classmethod
    def _from_user(cls, **kwargs):
        rules = []
        for k, v in kwargs.items():
            subcls = _FlowSpecComponentBase.lookup_type_name(k, cls.ROUTE_FAMILY.afi)
            rule = subcls.from_str(str(v))
            rules.extend(rule)
        rules.sort(key=lambda x: x.type)
        return cls(rules=rules)

    @property
    def prefix(self):

        def _format(i):
            pairs = []
            i.rules.sort(key=lambda x: x.type)
            previous_type = None
            for r in i.rules:
                if r.type == previous_type:
                    if r.to_str()[0] != '&':
                        pairs[-1] += '|'
                    pairs[-1] += r.to_str()
                else:
                    pairs.append('%s:%s' % (r.COMPONENT_NAME, r.to_str()))
                previous_type = r.type
            return ','.join(pairs)
        return '%s(%s)' % (self.FLOWSPEC_FAMILY, _format(self))

    @property
    def formatted_nlri_str(self):
        return self.prefix