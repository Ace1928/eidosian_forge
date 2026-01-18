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
@EvpnNLRI.register_type(EvpnNLRI.ETHERNET_AUTO_DISCOVERY)
class EvpnEthernetAutoDiscoveryNLRI(EvpnNLRI):
    """
    Ethernet A-D route type specific EVPN NLRI
    """
    ROUTE_TYPE_NAME = 'eth_ad'
    _PACK_STR = '!8s10sI3s'
    NLRI_PREFIX_FIELDS = ['esi', 'ethernet_tag_id']
    _TYPE = {'ascii': ['route_dist']}

    def __init__(self, route_dist, esi, ethernet_tag_id, mpls_label=None, vni=None, label=None, type_=None, length=None):
        super(EvpnEthernetAutoDiscoveryNLRI, self).__init__(type_, length)
        self.route_dist = route_dist
        self.esi = esi
        self.ethernet_tag_id = ethernet_tag_id
        if label:
            self._label = label
            self._mpls_label, _, _ = self._mpls_label_from_bin(label)
            self._vni, _ = self._vni_from_bin(label)
        else:
            self._label = self._serialize_label(mpls_label, vni)
            self._mpls_label = mpls_label
            self._vni = vni

    def _serialize_label(self, mpls_label, vni):
        if mpls_label:
            return self._mpls_label_to_bin(mpls_label, is_bos=True)
        elif vni:
            return self._vni_to_bin(vni)
        else:
            return b'\x00' * 3

    @classmethod
    def parse_value(cls, buf):
        route_dist, rest = cls._rd_from_bin(buf)
        esi, rest = cls._esi_from_bin(rest)
        ethernet_tag_id, rest = cls._ethernet_tag_id_from_bin(rest)
        return {'route_dist': route_dist.formatted_str, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'label': rest}

    def serialize_value(self):
        route_dist = _RouteDistinguisher.from_str(self.route_dist)
        return struct.pack(self._PACK_STR, route_dist.serialize(), self.esi.serialize(), self.ethernet_tag_id, self._label)

    @property
    def mpls_label(self):
        return self._mpls_label

    @mpls_label.setter
    def mpls_label(self, mpls_label):
        self._label = self._mpls_label_to_bin(mpls_label, is_bos=True)
        self._mpls_label = mpls_label
        self._vni = None

    @property
    def vni(self):
        return self._vni

    @vni.setter
    def vni(self, vni):
        self._label = self._vni_to_bin(vni)
        self._mpls_label = None
        self._vni = vni

    @property
    def label_list(self):
        return [self.mpls_label]