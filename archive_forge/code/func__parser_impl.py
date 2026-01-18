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
def _parser_impl(cls, buf, from_zebra=False):
    buf = bytes(buf)
    length, version, vrf_id, command, body_buf = cls.parse_header(buf)
    if body_buf:
        body_cls = cls.get_body_class(version, command)
        if from_zebra:
            body = body_cls.parse_from_zebra(body_buf, version=version)
        else:
            body = body_cls.parse(body_buf, version=version)
    else:
        body = None
    rest = buf[length:]
    if from_zebra:
        return (cls(length, version, vrf_id, command, body), _ZebraMessageFromZebra, rest)
    return (cls(length, version, vrf_id, command, body), cls, rest)