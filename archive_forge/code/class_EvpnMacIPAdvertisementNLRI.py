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
@EvpnNLRI.register_type(EvpnNLRI.MAC_IP_ADVERTISEMENT)
class EvpnMacIPAdvertisementNLRI(EvpnNLRI):
    """
    MAC/IP Advertisement route type specific EVPN NLRI
    """
    ROUTE_TYPE_NAME = 'mac_ip_adv'
    _PACK_STR = '!8s10sIB6sB%ds%ds'
    NLRI_PREFIX_FIELDS = ['ethernet_tag_id', 'mac_addr', 'ip_addr']
    _TYPE = {'ascii': ['route_dist', 'mac_addr', 'ip_addr']}

    def __init__(self, route_dist, ethernet_tag_id, mac_addr, ip_addr, esi=None, mpls_labels=None, vni=None, labels=None, mac_addr_len=None, ip_addr_len=None, type_=None, length=None):
        super(EvpnMacIPAdvertisementNLRI, self).__init__(type_, length)
        self.route_dist = route_dist
        self.esi = esi
        self.ethernet_tag_id = ethernet_tag_id
        self.mac_addr_len = mac_addr_len
        self.mac_addr = mac_addr
        self.ip_addr_len = ip_addr_len
        self.ip_addr = ip_addr
        if labels:
            self._mpls_labels, self._vni = self._parse_labels(labels)
            self._labels = labels
        else:
            self._labels = self._serialize_labels(mpls_labels, vni)
            self._mpls_labels = mpls_labels
            self._vni = vni

    def _parse_labels(self, labels):
        mpls_label1, rest, is_bos = self._mpls_label_from_bin(labels)
        mpls_labels = [mpls_label1]
        if rest and (not is_bos):
            mpls_label2, rest, _ = self._mpls_label_from_bin(rest)
            mpls_labels.append(mpls_label2)
        vni, _ = self._vni_from_bin(labels)
        return (mpls_labels, vni)

    def _serialize_labels(self, mpls_labels, vni):
        if mpls_labels:
            return self._serialize_mpls_labels(mpls_labels)
        elif vni:
            return self._vni_to_bin(vni)
        else:
            return b'\x00' * 3

    def _serialize_mpls_labels(self, mpls_labels):
        if len(mpls_labels) == 1:
            return self._mpls_label_to_bin(mpls_labels[0], is_bos=True)
        elif len(mpls_labels) == 2:
            return self._mpls_label_to_bin(mpls_labels[0], is_bos=False) + self._mpls_label_to_bin(mpls_labels[1], is_bos=True)
        else:
            return b'\x00' * 3

    @classmethod
    def parse_value(cls, buf):
        route_dist, rest = cls._rd_from_bin(buf)
        esi, rest = cls._esi_from_bin(rest)
        ethernet_tag_id, rest = cls._ethernet_tag_id_from_bin(rest)
        mac_addr_len, rest = cls._mac_addr_len_from_bin(rest)
        mac_addr, rest = cls._mac_addr_from_bin(rest, mac_addr_len)
        ip_addr_len, rest = cls._ip_addr_len_from_bin(rest)
        if ip_addr_len != 0:
            ip_addr, rest = cls._ip_addr_from_bin(rest, ip_addr_len // 8)
        else:
            ip_addr = None
        return {'route_dist': route_dist.formatted_str, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'mac_addr_len': mac_addr_len, 'mac_addr': mac_addr, 'ip_addr_len': ip_addr_len, 'ip_addr': ip_addr, 'labels': rest}

    def serialize_value(self):
        route_dist = _RouteDistinguisher.from_str(self.route_dist)
        mac_addr = self._mac_addr_to_bin(self.mac_addr)
        self.mac_addr_len = len(mac_addr) * 8
        if self.ip_addr:
            ip_addr = self._ip_addr_to_bin(self.ip_addr)
        else:
            ip_addr = b''
        ip_addr_len = len(ip_addr)
        self.ip_addr_len = ip_addr_len * 8
        return struct.pack(self._PACK_STR % (ip_addr_len, len(self._labels)), route_dist.serialize(), self.esi.serialize(), self.ethernet_tag_id, self.mac_addr_len, mac_addr, self.ip_addr_len, ip_addr, self._labels)

    @property
    def mpls_labels(self):
        return self._mpls_labels

    @mpls_labels.setter
    def mpls_labels(self, mpls_labels):
        self._labels = self._serialize_mpls_labels(mpls_labels)
        self._mpls_labels = mpls_labels
        self._vni = None

    @property
    def vni(self):
        return self._vni

    @vni.setter
    def vni(self, vni):
        self._labels = self._vni_to_bin(vni)
        self._mpls_labels = None
        self._vni = vni

    @property
    def label_list(self):
        return self.mpls_labels