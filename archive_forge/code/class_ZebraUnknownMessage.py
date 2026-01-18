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
@_FrrZebraMessageBody.register_unknown_type()
@_ZebraMessageBody.register_unknown_type()
class ZebraUnknownMessage(_ZebraMessageBody):
    """
    Message body class for Unknown command.
    """

    def __init__(self, buf):
        super(ZebraUnknownMessage, self).__init__()
        self.buf = buf

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        return cls(buf)

    def serialize(self, version=_DEFAULT_VERSION):
        return self.buf