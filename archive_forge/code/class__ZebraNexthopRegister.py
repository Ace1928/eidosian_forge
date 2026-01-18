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
class _ZebraNexthopRegister(_ZebraMessageBody):
    """
    Base class for ZEBRA_NEXTHOP_REGISTER and ZEBRA_NEXTHOP_UNREGISTER
    message body.
    """

    def __init__(self, nexthops):
        super(_ZebraNexthopRegister, self).__init__()
        nexthops = nexthops or []
        for nexthop in nexthops:
            assert isinstance(nexthop, RegisteredNexthop)
        self.nexthops = nexthops

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        nexthops = []
        while buf:
            nexthop, buf = RegisteredNexthop.parse(buf)
            nexthops.append(nexthop)
        return cls(nexthops)

    def serialize(self, version=_DEFAULT_VERSION):
        buf = b''
        for nexthop in self.nexthops:
            buf += nexthop.serialize()
        return buf