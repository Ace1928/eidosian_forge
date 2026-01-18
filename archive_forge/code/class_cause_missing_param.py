import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_abort.register_cause_code
@chunk_error.register_cause_code
class cause_missing_param(cause):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Missing Mandatory Parameter (RFC 4960).

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
    types          a list of missing params.
    num            Number of missing params.
                   (0 means automatically-calculate when encoding)
    length         length of this cause containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _PACK_STR = '!HHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def cause_code(cls):
        return CCODE_MISSING_PARAM

    def __init__(self, types=None, num=0, length=0):
        super(cause_missing_param, self).__init__(length)
        types = types or []
        assert isinstance(types, list)
        for one in types:
            assert isinstance(one, int)
        self.types = types
        self.num = num

    @classmethod
    def parser(cls, buf):
        _, length, num = struct.unpack_from(cls._PACK_STR, buf)
        types = []
        offset = cls._MIN_LEN
        for count in range(num):
            offset = cls._MIN_LEN + struct.calcsize('!H') * count
            one, = struct.unpack_from('!H', buf, offset)
            types.append(one)
        return cls(types, num, length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.cause_code(), self.length, self.num))
        for one in self.types:
            buf.extend(struct.pack('!H', one))
        if 0 == self.num:
            self.num = len(self.types)
            struct.pack_into('!I', buf, 4, self.num)
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)