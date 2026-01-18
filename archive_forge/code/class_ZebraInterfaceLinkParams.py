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
@_FrrZebraMessageBody.register_type(FRR_ZEBRA_INTERFACE_LINK_PARAMS)
@_ZebraMessageBody.register_type(ZEBRA_INTERFACE_LINK_PARAMS)
class ZebraInterfaceLinkParams(_ZebraMessageBody):
    """
    Message body class for ZEBRA_INTERFACE_LINK_PARAMS.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, ifindex, link_params):
        super(ZebraInterfaceLinkParams, self).__init__()
        self.ifindex = ifindex
        assert isinstance(link_params, InterfaceLinkParams)
        self.link_params = link_params

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        ifindex, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        link_params, rest = InterfaceLinkParams.parse(rest)
        return cls(ifindex, link_params)

    def serialize(self, version=_DEFAULT_VERSION):
        buf = struct.pack(self._HEADER_FMT, self.ifindex)
        return buf + self.link_params.serialize()