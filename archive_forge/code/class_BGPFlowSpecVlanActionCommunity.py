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
@_ExtendedCommunity.register_type(_ExtendedCommunity.FLOWSPEC_VLAN_ACTION)
class BGPFlowSpecVlanActionCommunity(_ExtendedCommunity):
    """
    Flow Specification Vlan Actions.

    ========= ===============================================
    Attribute Description
    ========= ===============================================
    actions_1 Bit representation of actions.
              Supported actions are
              ``POP``, ``PUSH``, ``SWAP``, ``REWRITE_INNER``, ``REWRITE_OUTER``.
    actions_2 Same as ``actions_1``.
    vlan_1    VLAN ID used by ``actions_1``.
    cos_1     Class of Service used by ``actions_1``.
    vlan_2    VLAN ID used by ``actions_2``.
    cos_2     Class of Service used by ``actions_2``.
    ========= ===============================================
    """
    _VALUE_PACK_STR = '!BBBHH'
    _VALUE_FIELDS = ['subtype', 'actions_1', 'actions_2', 'vlan_1', 'vlan_2', 'cos_1', 'cos_2']
    ACTION_NAME = 'vlan_action'
    _COS_MASK = 7
    POP = 1 << 7
    PUSH = 1 << 6
    SWAP = 1 << 5
    REWRITE_INNER = 1 << 4
    REWRITE_OUTER = 1 << 3

    def __init__(self, **kwargs):
        super(BGPFlowSpecVlanActionCommunity, self).__init__()
        kwargs['subtype'] = self.SUBTYPE_FLOWSPEC_VLAN_ACTION
        self.do_init(BGPFlowSpecVlanActionCommunity, self, kwargs)

    @classmethod
    def parse_value(cls, buf):
        subtype, actions_1, actions_2, vlan_cos_1, vlan_cos_2 = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'subtype': subtype, 'actions_1': actions_1, 'vlan_1': int(vlan_cos_1 >> 4), 'cos_1': int(vlan_cos_1 >> 1 & cls._COS_MASK), 'actions_2': actions_2, 'vlan_2': int(vlan_cos_2 >> 4), 'cos_2': int(vlan_cos_2 >> 1 & cls._COS_MASK)}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, self.subtype, self.actions_1, self.actions_2, (self.vlan_1 << 4) + (self.cos_1 << 1), (self.vlan_2 << 4) + (self.cos_2 << 1))