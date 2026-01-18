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
def _serialize_family_prefix(self, prefix):
    if ip.valid_ipv4(prefix):
        family = socket.AF_INET
        prefix_addr, prefix_num = prefix.split('/')
        return (family, struct.pack(self._FAMILY_IPV4_PREFIX_FMT, family, addrconv.ipv4.text_to_bin(prefix_addr), int(prefix_num)))
    elif ip.valid_ipv6(prefix):
        family = socket.AF_INET6
        prefix_addr, prefix_num = prefix.split('/')
        return (family, struct.pack(self._FAMILY_IPV6_PREFIX_FMT, family, addrconv.ipv6.text_to_bin(prefix_addr), int(prefix_num)))
    raise ValueError('Invalid prefix: %s' % prefix)