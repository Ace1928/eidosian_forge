import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_init.register_param_type
class param_cookie_preserve(param):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Cookie Preservative Parameter (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_init

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    value          Suggested Cookie Life-Span Increment (msec).
    length         length of this param containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _PACK_STR = '!HHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def param_type(cls):
        return PTYPE_COOKIE_PRESERVE

    def __init__(self, value=0, length=0):
        super(param_cookie_preserve, self).__init__(value, length)

    @classmethod
    def parser(cls, buf):
        _, length, value = struct.unpack_from(cls._PACK_STR, buf)
        return cls(value, length)

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.param_type(), self.length, self.value)
        return buf