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
class FlowSpecIPv4NLRI(_FlowSpecNLRIBase):
    """
    Flow Specification NLRI class for IPv4 [RFC 5575]
    """
    ROUTE_FAMILY = RF_IPv4_FLOWSPEC
    FLOWSPEC_FAMILY = 'ipv4fs'

    @classmethod
    def from_user(cls, **kwargs):
        """
        Utility method for creating a NLRI instance.

        This function returns a NLRI instance from human readable format value.

        :param kwargs: The following arguments are available.

        =========== ============= ========= ==============================
        Argument    Value         Operator  Description
        =========== ============= ========= ==============================
        dst_prefix  IPv4 Prefix   Nothing   Destination Prefix.
        src_prefix  IPv4 Prefix   Nothing   Source Prefix.
        ip_proto    Integer       Numeric   IP Protocol.
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
                                            ``DF`` (Don't fragment),
                                            ``ISF`` (Is a fragment),
                                            ``FF`` (First fragment) and
                                            ``LF`` (Last fragment)
        =========== ============= ========= ==============================

        Example::

            >>> msg = bgp.FlowSpecIPv4NLRI.from_user(
            ...     dst_prefix='10.0.0.0/24',
            ...     src_prefix='20.0.0.1/24',
            ...     ip_proto=6,
            ...     port='80 | 8000',
            ...     dst_port='>9000 & <9050',
            ...     src_port='>=8500 & <=9000',
            ...     icmp_type=0,
            ...     icmp_code=6,
            ...     tcp_flags='SYN+ACK & !=URGENT',
            ...     packet_len=1000,
            ...     dscp='22 | 24',
            ...     fragment='LF | ==FF')
            >>>

        You can specify conditions with the following keywords.

        The following keywords can be used when the operator type is Numeric.

        ========== ============================================================
        Keyword    Description
        ========== ============================================================
        <          Less than comparison between data and value.
        <=         Less than or equal to comparison between data and value.
        >          Greater than comparison between data and value.
        >=         Greater than or equal to comparison between data and value.
        ==         Equality between data and value.
                   This operator can be omitted.
        ========== ============================================================

        The following keywords can be used when the operator type is Bitmask.

        ========== ================================================
        Keyword    Description
        ========== ================================================
        !=         Not equal operation.
        ==         Exact match operation if specified.
                   Otherwise partial match operation.
        `+`        Used for the summation of bitmask values.
                   (e.g., SYN+ACK)
        ========== ================================================

        You can combine the multiple conditions with the following operators.

        ========== =======================================
        Keyword    Description
        ========== =======================================
        `|`        Logical OR operation
        &          Logical AND operation
        ========== =======================================

        :return: A instance of FlowSpecVPNv4NLRI.
        """
        return cls._from_user(**kwargs)