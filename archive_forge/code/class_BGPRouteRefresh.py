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
@BGPMessage.register_type(BGP_MSG_ROUTE_REFRESH)
class BGPRouteRefresh(BGPMessage):
    """BGP-4 ROUTE REFRESH Message (RFC 2918) encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    marker                     Marker field.  Ignored when encoding.
    len                        Length field.  Ignored when encoding.
    type                       Type field.
    afi                        Address Family Identifier
    safi                       Subsequent Address Family Identifier
    ========================== ===============================================
    """
    _PACK_STR = '!HBB'
    _MIN_LEN = BGPMessage._HDR_LEN + struct.calcsize(_PACK_STR)

    def __init__(self, afi, safi, demarcation=0, type_=BGP_MSG_ROUTE_REFRESH, len_=None, marker=None):
        super(BGPRouteRefresh, self).__init__(marker=marker, len_=len_, type_=type_)
        self.afi = afi
        self.safi = safi
        self.demarcation = demarcation
        self.eor_sent = False

    @classmethod
    def parser(cls, buf):
        afi, demarcation, safi = struct.unpack_from(cls._PACK_STR, bytes(buf))
        return {'afi': afi, 'safi': safi, 'demarcation': demarcation}

    def serialize_tail(self):
        return bytearray(struct.pack(self._PACK_STR, self.afi, self.demarcation, self.safi))