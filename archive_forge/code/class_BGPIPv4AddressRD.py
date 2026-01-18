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
@_RouteDistinguisher.register_type(_RouteDistinguisher.IPV4_ADDRESS)
class BGPIPv4AddressRD(_RouteDistinguisher):
    _VALUE_PACK_STR = '!4sH'
    _VALUE_FIELDS = ['admin', 'assigned']
    _TYPE = {'ascii': ['admin']}

    def __init__(self, **kwargs):
        super(BGPIPv4AddressRD, self).__init__()
        self.do_init(BGPIPv4AddressRD, self, kwargs)

    @classmethod
    def parse_value(cls, buf):
        d_ = super(BGPIPv4AddressRD, cls).parse_value(buf)
        d_['admin'] = addrconv.ipv4.bin_to_text(d_['admin'])
        return d_

    def serialize_value(self):
        args = []
        for f in self._VALUE_FIELDS:
            v = getattr(self, f)
            if f == 'admin':
                v = bytes(addrconv.ipv4.text_to_bin(v))
            args.append(v)
        buf = bytearray()
        msg_pack_into(self._VALUE_PACK_STR, buf, 0, *args)
        return buf