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
class RegisteredNexthop(stringify.StringifyMixin):
    """
    Unit of ZEBRA_NEXTHOP_REGISTER message body.
    """
    _HEADER_FMT = '!?H'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, connected, family, prefix):
        super(RegisteredNexthop, self).__init__()
        self.connected = connected
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix

    @property
    def flags(self):
        return self.connected

    @flags.setter
    def flags(self, v):
        self.connected = v

    @classmethod
    def parse(cls, buf):
        connected, family = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        prefix, rest = _parse_ip_prefix(family, rest)
        return (cls(connected, family, prefix), rest)

    def serialize(self):
        buf = struct.pack(self._HEADER_FMT, self.connected, self.family)
        return buf + _serialize_ip_prefix(self.prefix)