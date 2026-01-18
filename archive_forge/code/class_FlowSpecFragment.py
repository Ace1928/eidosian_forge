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
@_FlowSpecComponentBase.register_type(_FlowSpecIPv4Component.TYPE_FRAGMENT, addr_family.IP)
class FlowSpecFragment(_FlowSpecBitmask):
    """Fragment for Flow Specification NLRI component

    Set the bitmask for operand format at value.
    The following values are supported.

    ========== ===============================================
    Attribute  Description
    ========== ===============================================
    LF         Last fragment
    FF         First fragment
    ISF        Is a fragment
    DF         Don't fragment
    ========== ===============================================
    """
    COMPONENT_NAME = 'fragment'
    LF = 1 << 3
    FF = 1 << 2
    ISF = 1 << 1
    DF = 1 << 0
    _bitmask_flags = collections.OrderedDict()
    _bitmask_flags[LF] = 'LF'
    _bitmask_flags[FF] = 'FF'
    _bitmask_flags[ISF] = 'ISF'
    _bitmask_flags[DF] = 'DF'