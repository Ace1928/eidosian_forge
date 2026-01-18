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
@_ExtendedCommunity.register_type(_ExtendedCommunity.FLOWSPEC_TRAFFIC_ACTION)
class BGPFlowSpecTrafficActionCommunity(_ExtendedCommunity):
    """
    Flow Specification Traffic Filtering Actions for Traffic Action.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    action                     Apply action.
                               The supported action are
                               ``SAMPLE`` and ``TERMINAL``.
    ========================== ===============================================
    """
    _VALUE_PACK_STR = '!B5xB'
    _VALUE_FIELDS = ['subtype', 'action']
    ACTION_NAME = 'traffic_action'
    SAMPLE = 1 << 1
    TERMINAL = 1 << 0

    def __init__(self, **kwargs):
        super(BGPFlowSpecTrafficActionCommunity, self).__init__()
        kwargs['subtype'] = self.SUBTYPE_FLOWSPEC_TRAFFIC_ACTION
        self.do_init(BGPFlowSpecTrafficActionCommunity, self, kwargs)