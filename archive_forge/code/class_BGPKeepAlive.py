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
@BGPMessage.register_type(BGP_MSG_KEEPALIVE)
class BGPKeepAlive(BGPMessage):
    """BGP-4 KEEPALIVE Message encoder/decoder class.

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
    ========================== ===============================================
    """
    _MIN_LEN = BGPMessage._HDR_LEN

    def __init__(self, type_=BGP_MSG_KEEPALIVE, len_=None, marker=None):
        super(BGPKeepAlive, self).__init__(marker=marker, len_=len_, type_=type_)

    @classmethod
    def parser(cls, buf):
        return {}

    def serialize_tail(self):
        return bytearray()