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
@BGPMessage.register_type(BGP_MSG_NOTIFICATION)
class BGPNotification(BGPMessage):
    """BGP-4 NOTIFICATION Message encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    marker                     Marker field.  Ignored when encoding.
    len                        Length field.  Ignored when encoding.
    type                       Type field.
    error_code                 Error code field.
    error_subcode              Error subcode field.
    data                       Data field.
    ========================== ===============================================
    """
    _PACK_STR = '!BB'
    _MIN_LEN = BGPMessage._HDR_LEN + struct.calcsize(_PACK_STR)
    _REASONS = {(1, 1): 'Message Header Error: not synchronised', (1, 2): 'Message Header Error: bad message len', (1, 3): 'Message Header Error: bad message type', (2, 1): 'Open Message Error: unsupported version', (2, 2): 'Open Message Error: bad peer AS', (2, 3): 'Open Message Error: bad BGP identifier', (2, 4): 'Open Message Error: unsupported optional param', (2, 5): 'Open Message Error: authentication failure', (2, 6): 'Open Message Error: unacceptable hold time', (2, 7): 'Open Message Error: Unsupported Capability', (2, 8): 'Open Message Error: Unassigned', (3, 1): 'Update Message Error: malformed attribute list', (3, 2): 'Update Message Error: unrecognized well-known attr', (3, 3): 'Update Message Error: missing well-known attr', (3, 4): 'Update Message Error: attribute flags error', (3, 5): 'Update Message Error: attribute length error', (3, 6): 'Update Message Error: invalid origin attr', (3, 7): 'Update Message Error: as routing loop', (3, 8): 'Update Message Error: invalid next hop attr', (3, 9): 'Update Message Error: optional attribute error', (3, 10): 'Update Message Error: invalid network field', (3, 11): 'Update Message Error: malformed AS_PATH', (4, 1): 'Hold Timer Expired', (5, 1): 'Finite State Machine Error', (6, 1): 'Cease: Maximum Number of Prefixes Reached', (6, 2): 'Cease: Administrative Shutdown', (6, 3): 'Cease: Peer De-configured', (6, 4): 'Cease: Administrative Reset', (6, 5): 'Cease: Connection Rejected', (6, 6): 'Cease: Other Configuration Change', (6, 7): 'Cease: Connection Collision Resolution', (6, 8): 'Cease: Out of Resources'}

    def __init__(self, error_code, error_subcode, data=b'', type_=BGP_MSG_NOTIFICATION, len_=None, marker=None):
        super(BGPNotification, self).__init__(marker=marker, len_=len_, type_=type_)
        self.error_code = error_code
        self.error_subcode = error_subcode
        self.data = data

    @classmethod
    def parser(cls, buf):
        error_code, error_subcode = struct.unpack_from(cls._PACK_STR, bytes(buf))
        data = bytes(buf[2:])
        return {'error_code': error_code, 'error_subcode': error_subcode, 'data': data}

    def serialize_tail(self):
        msg = bytearray(struct.pack(self._PACK_STR, self.error_code, self.error_subcode))
        msg += self.data
        return msg

    @property
    def reason(self):
        return self._REASONS.get((self.error_code, self.error_subcode))