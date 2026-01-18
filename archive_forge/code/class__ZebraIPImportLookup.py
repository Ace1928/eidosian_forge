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
class _ZebraIPImportLookup(_ZebraMessageBody, metaclass=abc.ABCMeta):
    """
    Base class for ZEBRA_IPV4_IMPORT_LOOKUP and
    ZEBRA_IPV6_IMPORT_LOOKUP message body.

    .. Note::

        Zebra IPv4/v6 Import Lookup message have asymmetric structure.
        If the message sent from Zebra Daemon, set 'from_zebra=True' to
        create an instance of this class.
    """
    _PREFIX_LEN_FMT = '!B'
    PREFIX_LEN_SIZE = struct.calcsize(_PREFIX_LEN_FMT)
    _METRIC_FMT = '!I'
    METRIC_SIZE = struct.calcsize(_METRIC_FMT)
    PREFIX_CLS = None
    PREFIX_LEN = None

    def __init__(self, prefix, metric=None, nexthops=None, from_zebra=False):
        super(_ZebraIPImportLookup, self).__init__()
        if not from_zebra:
            assert ip.valid_ipv4(prefix) or ip.valid_ipv6(prefix)
        elif isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        else:
            assert ip.valid_ipv4(prefix) or ip.valid_ipv6(prefix)
        self.prefix = prefix
        self.metric = metric
        nexthops = nexthops or []
        for nexthop in nexthops:
            assert isinstance(nexthop, _NextHop)
        self.nexthops = nexthops
        self.from_zebra = from_zebra

    @classmethod
    def parse_impl(cls, buf, version=_DEFAULT_VERSION, from_zebra=False):
        if not from_zebra:
            prefix_len, = struct.unpack_from(cls._PREFIX_LEN_FMT, buf)
            rest = buf[cls.PREFIX_LEN_SIZE:]
            prefix = cls.PREFIX_CLS.bin_to_text(rest[:cls.PREFIX_LEN])
            return cls('%s/%d' % (prefix, prefix_len), from_zebra=False)
        prefix = cls.PREFIX_CLS.bin_to_text(buf[:cls.PREFIX_LEN])
        rest = buf[4:]
        metric, = struct.unpack_from(cls._METRIC_FMT, rest)
        rest = rest[cls.METRIC_SIZE:]
        nexthops, rest = _parse_nexthops(rest, version)
        return cls(prefix, metric, nexthops, from_zebra=True)

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        return cls.parse_impl(buf, version=version, from_zebra=False)

    @classmethod
    def parse_from_zebra(cls, buf, version=_DEFAULT_VERSION):
        return cls.parse_impl(buf, version=version, from_zebra=True)

    def serialize(self, version=_DEFAULT_VERSION):
        if not self.from_zebra:
            if ip.valid_ipv4(self.prefix) or ip.valid_ipv6(self.prefix):
                prefix, prefix_len = self.prefix.split('/')
                return struct.pack(self._PREFIX_LEN_FMT, int(prefix_len)) + self.PREFIX_CLS.text_to_bin(prefix)
            else:
                raise ValueError('Invalid prefix: %s' % self.prefix)
        if ip.valid_ipv4(self.prefix) or ip.valid_ipv6(self.prefix):
            buf = self.PREFIX_CLS.text_to_bin(self.prefix)
        else:
            raise ValueError('Invalid prefix: %s' % self.prefix)
        buf += struct.pack(self._METRIC_FMT, self.metric)
        return buf + _serialize_nexthops(self.nexthops, version=version)