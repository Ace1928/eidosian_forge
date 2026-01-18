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
@_ExtendedCommunity.register_type(_ExtendedCommunity.FLOWSPEC_TPID_ACTION)
class BGPFlowSpecTPIDActionCommunity(_ExtendedCommunity):
    """
    Flow Specification TPID Actions.

    ========= =========================================================
    Attribute Description
    ========= =========================================================
    actions   Bit representation of actions.
              Supported actions are
              ``TI(inner TPID action)`` and ``TO(outer TPID action)``.
    tpid_1    TPID used by ``TI``.
    tpid_2    TPID used by ``TO``.
    ========= =========================================================
    """
    _VALUE_PACK_STR = '!BHHH'
    _VALUE_FIELDS = ['subtype', 'actions', 'tpid_1', 'tpid_2']
    ACTION_NAME = 'tpid_action'
    TI = 1 << 15
    TO = 1 << 14

    def __init__(self, **kwargs):
        super(BGPFlowSpecTPIDActionCommunity, self).__init__()
        kwargs['subtype'] = self.SUBTYPE_FLOWSPEC_TPID_ACTION
        self.do_init(BGPFlowSpecTPIDActionCommunity, self, kwargs)

    @classmethod
    def parse_value(cls, buf):
        subtype, actions, tpid_1, tpid_2 = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'subtype': subtype, 'actions': actions, 'tpid_1': tpid_1, 'tpid_2': tpid_2}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, self.subtype, self.actions, self.tpid_1, self.tpid_2)