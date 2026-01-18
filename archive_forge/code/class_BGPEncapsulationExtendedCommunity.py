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
@_ExtendedCommunity.register_type(_ExtendedCommunity.ENCAPSULATION)
class BGPEncapsulationExtendedCommunity(_ExtendedCommunity):
    _VALUE_PACK_STR = '!B4xH'
    _VALUE_FIELDS = ['subtype', 'tunnel_type']
    TUNNEL_TYPE_L2TPV3 = 1
    TUNNEL_TYPE_GRE = 2
    TUNNEL_TYPE_IP_IN_IP = 7
    TUNNEL_TYPE_VXLAN = 8
    TUNNEL_TYPE_NVGRE = 9
    TUNNEL_TYPE_MPLS = 10
    TUNNEL_TYPE_MPLS_IN_GRE = 11
    TUNNEL_TYPE_VXLAN_GRE = 12
    TUNNEL_TYPE_MPLS_IN_UDP = 13

    def __init__(self, **kwargs):
        super(BGPEncapsulationExtendedCommunity, self).__init__()
        self.do_init(BGPEncapsulationExtendedCommunity, self, kwargs)

    @classmethod
    def from_str(cls, tunnel_type):
        """
        Returns an instance identified with the given `tunnel_type`.

        `tunnel_type` should be a str type value and corresponding to
        BGP Tunnel Encapsulation Attribute Tunnel Type constants name
        omitting `TUNNEL_TYPE_` prefix.

        Example:
            - `gre` means TUNNEL_TYPE_GRE
            - `vxlan` means TUNNEL_TYPE_VXLAN

        And raises AttributeError when the corresponding Tunnel Type
        is not found to the given `tunnel_type`.
        """
        return cls(subtype=_ExtendedCommunity.SUBTYPE_ENCAPSULATION, tunnel_type=getattr(cls, 'TUNNEL_TYPE_' + tunnel_type.upper()))