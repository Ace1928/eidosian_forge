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
@ipv6.register_header_type(inet.IPPROTO_AH)
class auth(header):
    """IP Authentication header (RFC 2402) encoder/decoder class.

    This is used with os_ken.lib.packet.ipv6.ipv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    nxt            Next Header
    size           the length of the Authentication Header
                   in 64-bit words, subtracting 1.
    spi            security parameters index.
    seq            sequence number.
    data           authentication data.
    ============== =======================================
    """
    TYPE = inet.IPPROTO_AH
    _PACK_STR = '!BB2xII'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, nxt=inet.IPPROTO_TCP, size=2, spi=0, seq=0, data=b'\x00\x00\x00\x00'):
        super(auth, self).__init__(nxt)
        assert data is not None
        self.size = size
        self.spi = spi
        self.seq = seq
        self.data = data

    @classmethod
    def _get_size(cls, size):
        return (int(size) + 2) * 4

    @classmethod
    def parser(cls, buf):
        nxt, size, spi, seq = struct.unpack_from(cls._PACK_STR, buf)
        form = '%ds' % (cls._get_size(size) - cls._MIN_LEN)
        data, = struct.unpack_from(form, buf, cls._MIN_LEN)
        return cls(nxt, size, spi, seq, data)

    def serialize(self):
        buf = struct.pack(self._PACK_STR, self.nxt, self.size, self.spi, self.seq)
        buf = bytearray(buf)
        form = '%ds' % (auth._get_size(self.size) - self._MIN_LEN)
        buf.extend(struct.pack(form, self.data))
        return buf

    def __len__(self):
        return auth._get_size(self.size)