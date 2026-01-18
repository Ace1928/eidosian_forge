import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_abort.register_cause_code
@chunk_error.register_cause_code
class cause_invalid_stream_id(cause_with_value):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Invalid Stream Identifier (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_abort
    - os_ken.lib.packet.sctp.chunk_error

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    value          stream id.
    length         length of this cause containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _PACK_STR = '!HHH2x'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def cause_code(cls):
        return CCODE_INVALID_STREAM_ID

    def __init__(self, value=0, length=0):
        super(cause_invalid_stream_id, self).__init__(value, length)

    @classmethod
    def parser(cls, buf):
        _, length, value = struct.unpack_from(cls._PACK_STR, buf)
        return cls(value, length)

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.cause_code(), self.length, self.value)
        return buf