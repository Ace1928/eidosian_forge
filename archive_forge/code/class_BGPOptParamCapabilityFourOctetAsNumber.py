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
@_OptParamCapability.register_type(BGP_CAP_FOUR_OCTET_AS_NUMBER)
class BGPOptParamCapabilityFourOctetAsNumber(_OptParamCapability):
    _CAP_PACK_STR = '!I'

    def __init__(self, as_number, **kwargs):
        super(BGPOptParamCapabilityFourOctetAsNumber, self).__init__(**kwargs)
        self.as_number = as_number

    @classmethod
    def parse_cap_value(cls, buf):
        as_number, = struct.unpack_from(cls._CAP_PACK_STR, bytes(buf))
        return {'as_number': as_number}

    def serialize_cap_value(self):
        buf = bytearray()
        msg_pack_into(self._CAP_PACK_STR, buf, 0, self.as_number)
        return buf