import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_shutdown_complete(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Shutdown Complete (SHUTDOWN COMPLETE)
    chunk (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    tflag          '0' means the Verification tag is normal. '1' means
                   the Verification tag is copy of the sender.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _PACK_STR = '!BBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def chunk_type(cls):
        return TYPE_SHUTDOWN_COMPLETE

    def __init__(self, tflag=0, length=0):
        assert 1 == tflag | 1
        super(chunk_shutdown_complete, self).__init__(self.chunk_type(), length)
        self.tflag = tflag

    @classmethod
    def parser(cls, buf):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        tflag = flags & 1
        msg = cls(tflag, length)
        return msg

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.chunk_type(), self.tflag, self.length)
        return buf