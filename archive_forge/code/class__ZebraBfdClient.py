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
class _ZebraBfdClient(_ZebraMessageBody):
    """
    Base class for FRR_ZEBRA_BFD_CLIENT_*.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, pid):
        super(_ZebraBfdClient, self).__init__()
        self.pid = pid

    @classmethod
    def parse(cls, buf, version=_DEFAULT_FRR_VERSION):
        pid, = struct.unpack_from(cls._HEADER_FMT, buf)
        return cls(pid)

    def serialize(self, version=_DEFAULT_FRR_VERSION):
        return struct.pack(self._HEADER_FMT, self.pid)