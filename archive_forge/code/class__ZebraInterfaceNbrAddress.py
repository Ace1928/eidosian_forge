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
class _ZebraInterfaceNbrAddress(_ZebraMessageBody):
    """
    Base class for FRR_ZEBRA_INTERFACE_NBR_ADDRESS_* message body.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, ifindex, family, prefix):
        super(_ZebraInterfaceNbrAddress, self).__init__()
        self.ifindex = ifindex
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        ifindex, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        family, prefix, _ = _parse_zebra_family_prefix(rest)
        return cls(ifindex, family, prefix)

    def serialize(self, version=_DEFAULT_VERSION):
        self.family, body_bin = _serialize_zebra_family_prefix(self.prefix)
        return struct.pack(self._HEADER_FMT, self.ifindex) + body_bin