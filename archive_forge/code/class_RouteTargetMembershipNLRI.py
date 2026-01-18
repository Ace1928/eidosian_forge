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
@functools.total_ordering
class RouteTargetMembershipNLRI(StringifyMixin):
    """Route Target Membership NLRI.

    Route Target membership NLRI is advertised in BGP UPDATE messages using
    the MP_REACH_NLRI and MP_UNREACH_NLRI attributes.
    """
    ROUTE_FAMILY = RF_RTC_UC
    DEFAULT_AS = '0:0'
    DEFAULT_RT = '0:0'

    def __init__(self, origin_as, route_target):
        if not (origin_as is self.DEFAULT_AS and route_target is self.DEFAULT_RT):
            if not self._is_valid_asn(origin_as) or not self._is_valid_ext_comm_attr(route_target):
                raise ValueError('Invalid params.')
        self.origin_as = origin_as
        self.route_target = route_target

    def _is_valid_asn(self, asn):
        """Returns True if the given AS number is Two or Four Octet."""
        if isinstance(asn, int) and 0 <= asn <= 4294967295:
            return True
        else:
            return False

    def _is_valid_ext_comm_attr(self, attr):
        """Validates *attr* as string representation of RT or SOO.

        Returns True if *attr* is as per our convention of RT or SOO, else
        False. Our convention is to represent RT/SOO is a string with format:
        *global_admin_part:local_admin_path*
        """
        is_valid = True
        if not isinstance(attr, str):
            is_valid = False
        else:
            first, second = attr.split(':')
            try:
                if '.' in first:
                    socket.inet_aton(first)
                else:
                    int(first)
                    int(second)
            except (ValueError, socket.error):
                is_valid = False
        return is_valid

    @property
    def formatted_nlri_str(self):
        return '%s:%s' % (self.origin_as, self.route_target)

    def is_default_rtnlri(self):
        if self._origin_as is self.DEFAULT_AS and self._route_target is self.DEFAULT_RT:
            return True
        return False

    def __lt__(self, other):
        return (self.origin_as, self.route_target) < (other.origin_as, other.route_target)

    def __eq__(self, other):
        return (self.origin_as, self.route_target) == (other.origin_as, other.route_target)

    def __hash__(self):
        return hash((self.origin_as, self.route_target))

    @classmethod
    def parser(cls, buf):
        idx = 0
        origin_as, = struct.unpack_from('!I', buf, idx)
        idx += 4
        route_target = _ExtendedCommunity(buf[idx:])
        return cls(origin_as, route_target)

    def serialize(self):
        rt_nlri = b''
        if not self.is_default_rtnlri():
            rt_nlri += struct.pack('!I', self.origin_as)
            rt_nlri += self.route_target.serialize()
        return struct.pack('B', 8 * 12) + rt_nlri