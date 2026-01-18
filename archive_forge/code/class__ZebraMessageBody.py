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
class _ZebraMessageBody(type_desc.TypeDisp, stringify.StringifyMixin):
    """
    Base class for Zebra message body.
    """

    @classmethod
    def lookup_command(cls, command):
        return cls._lookup_type(command)

    @classmethod
    def rev_lookup_command(cls, body_cls):
        return cls._rev_lookup_type(body_cls)

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        return cls()

    @classmethod
    def parse_from_zebra(cls, buf, version=_DEFAULT_VERSION):
        return cls.parse(buf, version=version)

    def serialize(self, version=_DEFAULT_VERSION):
        return b''