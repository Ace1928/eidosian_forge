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
@_OptParamCapability.register_type(BGP_CAP_GRACEFUL_RESTART)
class BGPOptParamCapabilityGracefulRestart(_OptParamCapability):
    _CAP_PACK_STR = '!H'

    def __init__(self, flags, time, tuples, **kwargs):
        super(BGPOptParamCapabilityGracefulRestart, self).__init__(**kwargs)
        self.flags = flags
        self.time = time
        self.tuples = tuples

    @classmethod
    def parse_cap_value(cls, buf):
        restart, = struct.unpack_from(cls._CAP_PACK_STR, bytes(buf))
        buf = buf[2:]
        l = []
        while len(buf) >= 4:
            l.append(struct.unpack_from('!HBB', buf))
            buf = buf[4:]
        return {'flags': restart >> 12, 'time': restart & 4095, 'tuples': l}

    def serialize_cap_value(self):
        buf = bytearray()
        msg_pack_into(self._CAP_PACK_STR, buf, 0, self.flags << 12 | self.time)
        offset = 2
        for i in self.tuples:
            afi, safi, flags = i
            msg_pack_into('!HBB', buf, offset, afi, safi, flags)
            offset += 4
        return buf