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
@_FlowSpecComponentBase.register_type(_FlowSpecIPv4Component.TYPE_TCP_FLAGS, addr_family.IP)
@_FlowSpecComponentBase.register_type(_FlowSpecIPv6Component.TYPE_TCP_FLAGS, addr_family.IP6)
class FlowSpecTCPFlags(_FlowSpecBitmask):
    """TCP flags for Flow Specification NLRI component

    Supported TCP flags are CWR, ECN, URGENT, ACK, PUSH, RST, SYN and FIN.
    """
    COMPONENT_NAME = 'tcp_flags'
    CWR = 1 << 7
    ECN = 1 << 6
    URGENT = 1 << 5
    ACK = 1 << 4
    PUSH = 1 << 3
    RST = 1 << 2
    SYN = 1 << 1
    FIN = 1 << 0
    _bitmask_flags = collections.OrderedDict()
    _bitmask_flags[SYN] = 'SYN'
    _bitmask_flags[ACK] = 'ACK'
    _bitmask_flags[FIN] = 'FIN'
    _bitmask_flags[RST] = 'RST'
    _bitmask_flags[PUSH] = 'PUSH'
    _bitmask_flags[URGENT] = 'URGENT'
    _bitmask_flags[ECN] = 'ECN'
    _bitmask_flags[CWR] = 'CWR'