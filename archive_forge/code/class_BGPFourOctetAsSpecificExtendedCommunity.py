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
@_ExtendedCommunity.register_type(_ExtendedCommunity.FOUR_OCTET_AS_SPECIFIC)
class BGPFourOctetAsSpecificExtendedCommunity(_ExtendedCommunity):
    _VALUE_PACK_STR = '!BIH'
    _VALUE_FIELDS = ['subtype', 'as_number', 'local_administrator']

    def __init__(self, **kwargs):
        super(BGPFourOctetAsSpecificExtendedCommunity, self).__init__()
        self.do_init(BGPFourOctetAsSpecificExtendedCommunity, self, kwargs)