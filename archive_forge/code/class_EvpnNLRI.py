import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
class EvpnNLRI(StringifyMixin, TypeDisp):
    """
    BGP Network Layer Reachability Information (NLRI) for EVPN
    """
    ROUTE_FAMILY = RF_L2_EVPN
    _PACK_STR = '!BB'
    _PACK_STR_SIZE = struct.calcsize(_PACK_STR)
    ETHERNET_AUTO_DISCOVERY = 1
    MAC_IP_ADVERTISEMENT = 2
    INCLUSIVE_MULTICAST_ETHERNET_TAG = 3
    ETHERNET_SEGMENT = 4
    IP_PREFIX_ROUTE = 5
    ROUTE_TYPE_NAME = None
    MAX_ET = 4294967295
    _NAMES = {}
    NLRI_PREFIX_FIELDS = []

    def __init__(self, type_=None, length=None):
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_
        self.length = length
        self.route_dist = None

    @classmethod
    def register_type(cls, type_):
        cls._TYPES = cls._TYPES.copy()
        cls._NAMES = cls._NAMES.copy()

        def _register_type(subcls):
            cls._TYPES[type_] = subcls
            cls._NAMES[subcls.ROUTE_TYPE_NAME] = subcls
            cls._REV_TYPES = None
            return subcls
        return _register_type

    @classmethod
    def _lookup_type_name(cls, type_name):
        try:
            return cls._NAMES[type_name]
        except KeyError:
            return EvpnUnknownNLRI

    @classmethod
    def parser(cls, buf):
        route_type, length = struct.unpack_from(cls._PACK_STR, bytes(buf))
        offset = cls._PACK_STR_SIZE + length
        subcls = cls._lookup_type(route_type)
        values = subcls.parse_value(buf[cls._PACK_STR_SIZE:offset])
        return (subcls(type_=route_type, length=length, **values), buf[offset:])

    def serialize_value(self):
        return b''

    def serialize(self):
        value_bin = self.serialize_value()
        self.length = len(value_bin)
        return struct.pack(EvpnNLRI._PACK_STR, self.type, self.length) + value_bin

    @staticmethod
    def _rd_from_bin(buf):
        return (_RouteDistinguisher.parser(buf[:8]), buf[8:])

    @staticmethod
    def _rd_to_bin(rd):
        return bytes(rd.serialize())

    @staticmethod
    def _esi_from_bin(buf):
        return (EvpnEsi.parser(buf[:10]), buf[10:])

    @staticmethod
    def _esi_to_bin(esi):
        return esi.serialize()

    @staticmethod
    def _ethernet_tag_id_from_bin(buf):
        return (type_desc.Int4.to_user(bytes(buf[:4])), buf[4:])

    @staticmethod
    def _ethernet_tag_id_to_bin(tag_id):
        return type_desc.Int4.from_user(tag_id)

    @staticmethod
    def _mac_addr_len_from_bin(buf):
        return (type_desc.Int1.to_user(bytes(buf[:1])), buf[1:])

    @staticmethod
    def _mac_addr_len_to_bin(mac_len):
        return type_desc.Int1.from_user(mac_len)

    @staticmethod
    def _mac_addr_from_bin(buf, mac_len):
        mac_len //= 8
        return (addrconv.mac.bin_to_text(buf[:mac_len]), buf[mac_len:])

    @staticmethod
    def _mac_addr_to_bin(mac_addr):
        return addrconv.mac.text_to_bin(mac_addr)

    @staticmethod
    def _ip_addr_len_from_bin(buf):
        return (type_desc.Int1.to_user(bytes(buf[:1])), buf[1:])

    @staticmethod
    def _ip_addr_len_to_bin(ip_len):
        return type_desc.Int1.from_user(ip_len)

    @staticmethod
    def _ip_addr_from_bin(buf, ip_len):
        return (ip.bin_to_text(buf[:ip_len]), buf[ip_len:])

    @staticmethod
    def _ip_addr_to_bin(ip_addr):
        return ip.text_to_bin(ip_addr)

    @staticmethod
    def _mpls_label_from_bin(buf):
        mpls_label, is_bos = mpls.label_from_bin(buf)
        rest = buf[3:]
        return (mpls_label, rest, is_bos)

    @staticmethod
    def _mpls_label_to_bin(label, is_bos=True):
        return mpls.label_to_bin(label, is_bos=is_bos)

    @staticmethod
    def _vni_from_bin(buf):
        return (vxlan.vni_from_bin(bytes(buf[:3])), buf[3:])

    @staticmethod
    def _vni_to_bin(vni):
        return vxlan.vni_to_bin(vni)

    @property
    def prefix(self):

        def _format(i):
            pairs = []
            for k in i.NLRI_PREFIX_FIELDS:
                v = getattr(i, k)
                if k == 'esi':
                    pairs.append('%s:%s' % (k, v.formatted_str))
                else:
                    pairs.append('%s:%s' % (k, v))
            return ','.join(pairs)
        return '%s(%s)' % (self.ROUTE_TYPE_NAME, _format(self))

    @property
    def formatted_nlri_str(self):
        return '%s:%s' % (self.route_dist, self.prefix)