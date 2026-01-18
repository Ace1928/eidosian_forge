import abc
import struct
from . import packet_base
from . import icmpv6
from . import tcp
from . import udp
from . import sctp
from . import gre
from . import in_proto as inet
from os_ken.lib import addrconv
from os_ken.lib import stringify
@ipv6.register_header_type(inet.IPPROTO_DSTOPTS)
class dst_opts(opt_header):
    """IPv6 (RFC 2460) destination header encoder/decoder class.

    This is used with os_ken.lib.packet.ipv6.ipv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    nxt            Next Header
    size           the length of the destination header,
                   not include the first 8 octet.
    data           IPv6 options.
    ============== =======================================
    """
    TYPE = inet.IPPROTO_DSTOPTS

    def __init__(self, nxt=inet.IPPROTO_TCP, size=0, data=None):
        super(dst_opts, self).__init__(nxt, size, data)