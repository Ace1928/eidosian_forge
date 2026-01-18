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
@_ZebraMessageBody.register_type(ZEBRA_IPV4_IMPORT_LOOKUP)
class ZebraIPv4ImportLookup(_ZebraIPImportLookup):
    """
    Message body class for ZEBRA_IPV4_IMPORT_LOOKUP.
    """
    PREFIX_CLS = addrconv.ipv4
    PREFIX_LEN = 4