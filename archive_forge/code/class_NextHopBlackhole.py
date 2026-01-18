import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
@_FrrNextHop.register_type(FRR_ZEBRA_NEXTHOP_BLACKHOLE)
@_NextHop.register_type(ZEBRA_NEXTHOP_BLACKHOLE)
class NextHopBlackhole(_NextHop):
    """
    Nexthop class for ZEBRA_NEXTHOP_BLACKHOLE type.
    """

    @classmethod
    def parse(cls, buf):
        return (cls(), buf)

    def _serialize(self):
        return b''