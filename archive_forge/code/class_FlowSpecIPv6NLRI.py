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
class FlowSpecIPv6NLRI(_FlowSpecNLRIBase):
    """
    Flow Specification NLRI class for IPv6 [RFC draft-ietf-idr-flow-spec-v6-08]
    """
    ROUTE_FAMILY = RF_IPv6_FLOWSPEC
    FLOWSPEC_FAMILY = 'ipv6fs'

    @classmethod
    def from_user(cls, **kwargs):
        """
        Utility method for creating a NLRI instance.

        This function returns a NLRI instance from human readable format value.

        :param kwargs: The following arguments are available.

        =========== ============= ========= ==============================
        Argument    Value         Operator  Description
        =========== ============= ========= ==============================
        dst_prefix  IPv6 Prefix   Nothing   Destination Prefix.
        src_prefix  IPv6 Prefix   Nothing   Source Prefix.
        next_header Integer       Numeric   Next Header.
        port        Integer       Numeric   Port number.
        dst_port    Integer       Numeric   Destination port number.
        src_port    Integer       Numeric   Source port number.
        icmp_type   Integer       Numeric   ICMP type.
        icmp_code   Integer       Numeric   ICMP code.
        tcp_flags   Fixed string  Bitmask   TCP flags.
                                            Supported values are
                                            ``CWR``, ``ECN``, ``URGENT``,
                                            ``ACK``, ``PUSH``, ``RST``,
                                            ``SYN`` and ``FIN``.
        packet_len  Integer       Numeric   Packet length.
        dscp        Integer       Numeric   Differentiated Services
                                            Code Point.
        fragment    Fixed string  Bitmask   Fragment.
                                            Supported values are
                                            ``ISF`` (Is a fragment),
                                            ``FF`` (First fragment) and
                                            ``LF`` (Last fragment)
        flow_label   Intefer      Numeric   Flow Label.
        =========== ============= ========= ==============================

        .. Note::

            For ``dst_prefix`` and ``src_prefix``, you can give "offset" value
            like this: ``2001::2/128/32``. At this case, ``offset`` is 32.
            ``offset`` can be omitted, then ``offset`` is treated as 0.
        """
        return cls._from_user(**kwargs)