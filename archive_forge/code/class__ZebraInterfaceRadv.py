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
class _ZebraInterfaceRadv(_ZebraMessageBody):
    """
    Base class for FRR_ZEBRA_INTERFACE_*_RADV message body.
    """
    _HEADER_FMT = '!II'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, ifindex, interval):
        super(_ZebraInterfaceRadv, self).__init__()
        self.ifindex = ifindex
        self.interval = interval

    @classmethod
    def parse(cls, buf, version=_DEFAULT_FRR_VERSION):
        ifindex, interval = struct.unpack_from(cls._HEADER_FMT, buf)
        return cls(ifindex, interval)

    def serialize(self, version=_DEFAULT_FRR_VERSION):
        return struct.pack(self._HEADER_FMT, self.ifindex, self.interval)