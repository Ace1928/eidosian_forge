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
@classmethod
def _parse_message_option(cls, message, flag, fmt, buf):
    if message & flag:
        option, = struct.unpack_from(fmt, buf)
        return (option, buf[struct.calcsize(fmt):])
    return (None, buf)