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
@_FrrZebraMessageBody.register_type(FRR_ZEBRA_HELLO)
@_ZebraMessageBody.register_type(ZEBRA_HELLO)
class ZebraHello(_ZebraMessageBody):
    """
    Message body class for ZEBRA_HELLO.
    """
    _HEADER_FMT = '!B'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _V4_HEADER_FMT = '!BH'
    V4_HEADER_SIZE = struct.calcsize(_V4_HEADER_FMT)

    def __init__(self, route_type, instance=None):
        super(ZebraHello, self).__init__()
        self.route_type = route_type
        self.instance = instance

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        instance = None
        if version <= 3:
            route_type, = struct.unpack_from(cls._HEADER_FMT, buf)
        elif version == 4:
            route_type, instance = struct.unpack_from(cls._V4_HEADER_FMT, buf)
        else:
            raise struct.error('Unsupported Zebra protocol version: %d' % version)
        return cls(route_type, instance)

    def serialize(self, version=_DEFAULT_VERSION):
        if version <= 3:
            return struct.pack(self._HEADER_FMT, self.route_type)
        elif version == 4:
            return struct.pack(self._V4_HEADER_FMT, self.route_type, self.instance)
        else:
            raise ValueError('Unsupported Zebra protocol version: %d' % version)