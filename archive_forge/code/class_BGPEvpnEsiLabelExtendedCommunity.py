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
@_ExtendedCommunity.register_type(_ExtendedCommunity.EVPN_ESI_LABEL)
class BGPEvpnEsiLabelExtendedCommunity(_ExtendedCommunity):
    """
    ESI Label Extended Community
    """
    _VALUE_PACK_STR = '!BB2x3s'
    _VALUE_FIELDS = ['subtype', 'flags']
    SINGLE_ACTIVE_BIT = 1 << 0

    def __init__(self, label=None, mpls_label=None, vni=None, **kwargs):
        super(BGPEvpnEsiLabelExtendedCommunity, self).__init__()
        self.do_init(BGPEvpnEsiLabelExtendedCommunity, self, kwargs)
        if label:
            self._label = label
            self._mpls_label, _ = mpls.label_from_bin(label)
            self._vni = vxlan.vni_from_bin(label)
        else:
            self._label = self._serialize_label(mpls_label, vni)
            self._mpls_label = mpls_label
            self._vni = vni

    def _serialize_label(self, mpls_label, vni):
        if mpls_label:
            return mpls.label_to_bin(mpls_label, is_bos=True)
        elif vni:
            return vxlan.vni_to_bin(vni)
        else:
            return b'\x00' * 3

    @classmethod
    def parse_value(cls, buf):
        subtype, flags, label = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'subtype': subtype, 'flags': flags, 'label': label}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, self.subtype, self.flags, self._label)

    @property
    def mpls_label(self):
        return self._mpls_label

    @mpls_label.setter
    def mpls_label(self, mpls_label):
        self._label = mpls.label_to_bin(mpls_label, is_bos=True)
        self._mpls_label = mpls_label
        self._vni = None

    @property
    def vni(self):
        return self._vni

    @vni.setter
    def vni(self, vni):
        self._label = vxlan.vni_to_bin(vni)
        self._mpls_label = None
        self._vni = vni