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
class BadMsg(BgpExc):
    """Error to indicate un-recognized message type.

    RFC says: If the Type field of the message header is not recognized, then
    the Error Subcode MUST be set to Bad Message Type.  The Data field MUST
    contain the erroneous Type field.
    """
    CODE = BGP_ERROR_MESSAGE_HEADER_ERROR
    SUB_CODE = BGP_ERROR_SUB_BAD_MESSAGE_TYPE

    def __init__(self, msg_type):
        super(BadMsg, self).__init__()
        self.msg_type = msg_type
        self.data = struct.pack('B', msg_type)

    def __str__(self):
        return '<BadMsg %d>' % (self.msg_type,)