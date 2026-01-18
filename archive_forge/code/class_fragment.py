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
@ipv6.register_header_type(inet.IPPROTO_FRAGMENT)
class fragment(header):
    """IPv6 (RFC 2460) fragment header encoder/decoder class.

    This is used with os_ken.lib.packet.ipv6.ipv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    nxt            Next Header
    offset         offset, in 8-octet units, relative to
                   the start of the fragmentable part of
                   the original packet.
    more           1 means more fragments follow;
                   0 means last fragment.
    id\\_           packet identification value.
    ============== =======================================
    """
    TYPE = inet.IPPROTO_FRAGMENT
    _PACK_STR = '!BxHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, nxt=inet.IPPROTO_TCP, offset=0, more=0, id_=0):
        super(fragment, self).__init__(nxt)
        self.offset = offset
        self.more = more
        self.id_ = id_

    @classmethod
    def parser(cls, buf):
        nxt, off_m, id_ = struct.unpack_from(cls._PACK_STR, buf)
        offset = off_m >> 3
        more = off_m & 1
        return cls(nxt, offset, more, id_)

    def serialize(self):
        off_m = self.offset << 3 | self.more
        buf = struct.pack(self._PACK_STR, self.nxt, off_m, self.id_)
        return buf

    def __len__(self):
        return self._MIN_LEN