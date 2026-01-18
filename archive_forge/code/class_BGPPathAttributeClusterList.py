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
@_PathAttribute.register_type(BGP_ATTR_TYPE_CLUSTER_LIST)
class BGPPathAttributeClusterList(_PathAttribute):
    _VALUE_PACK_STR = '!4s'
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL
    _TYPE = {'ascii': ['value']}

    @classmethod
    def parse_value(cls, buf):
        rest = buf
        cluster_list = []
        elem_size = struct.calcsize(cls._VALUE_PACK_STR)
        while len(rest) >= elem_size:
            cluster_id, = struct.unpack_from(cls._VALUE_PACK_STR, bytes(rest))
            cluster_list.append(addrconv.ipv4.bin_to_text(cluster_id))
            rest = rest[elem_size:]
        return {'value': cluster_list}

    def serialize_value(self):
        buf = bytearray()
        offset = 0
        for cluster_id in self.value:
            msg_pack_into(self._VALUE_PACK_STR, buf, offset, addrconv.ipv4.text_to_bin(cluster_id))
            offset += struct.calcsize(self._VALUE_PACK_STR)
        return buf