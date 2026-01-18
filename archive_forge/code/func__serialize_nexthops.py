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
def _serialize_nexthops(nexthops, version=_DEFAULT_VERSION):
    nexthop_count = len(nexthops)
    buf = struct.pack(_NEXTHOP_COUNT_FMT, nexthop_count)
    if nexthop_count == 0:
        return buf
    for nexthop in nexthops:
        buf += nexthop.serialize(version=version)
    return buf