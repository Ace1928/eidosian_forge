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
@_OptParamCapability.register_type(BGP_CAP_MULTIPROTOCOL)
class BGPOptParamCapabilityMultiprotocol(_OptParamCapability):
    _CAP_PACK_STR = '!HBB'

    def __init__(self, afi, safi, reserved=0, **kwargs):
        super(BGPOptParamCapabilityMultiprotocol, self).__init__(**kwargs)
        self.afi = afi
        self.reserved = reserved
        self.safi = safi

    @classmethod
    def parse_cap_value(cls, buf):
        afi, reserved, safi = struct.unpack_from(cls._CAP_PACK_STR, bytes(buf))
        return {'afi': afi, 'reserved': reserved, 'safi': safi}

    def serialize_cap_value(self):
        self.reserved = 0
        buf = bytearray()
        msg_pack_into(self._CAP_PACK_STR, buf, 0, self.afi, self.reserved, self.safi)
        return buf