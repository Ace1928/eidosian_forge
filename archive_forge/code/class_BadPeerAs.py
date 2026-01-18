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
class BadPeerAs(BgpExc):
    """Error to indicate open message has incorrect AS number.

    RFC says: If the Autonomous System field of the OPEN message is
    unacceptable, then the Error Subcode MUST be set to Bad Peer AS.  The
    determination of acceptable Autonomous System numbers is configure peer AS.
    """
    CODE = BGP_ERROR_OPEN_MESSAGE_ERROR
    SUB_CODE = BGP_ERROR_SUB_BAD_PEER_AS