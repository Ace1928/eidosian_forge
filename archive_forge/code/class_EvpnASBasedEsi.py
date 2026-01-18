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
@EvpnEsi.register_type(EvpnEsi.AS_BASED)
class EvpnASBasedEsi(EvpnEsi):
    """
    AS based ESI value

    This type indicates an Autonomous System(AS)-based
    ESI Value that can be auto-generated or configured by
    the operator.
    """
    _TYPE_NAME = 'as_based'
    _VALUE_PACK_STR = '!IIx'
    _VALUE_FIELDS = ['as_number', 'local_disc']

    def __init__(self, as_number, local_disc, type_=None):
        super(EvpnASBasedEsi, self).__init__(type_)
        self.as_number = as_number
        self.local_disc = local_disc