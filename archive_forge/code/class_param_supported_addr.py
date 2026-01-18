import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_init.register_param_type
class param_supported_addr(param):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Supported Address Types Parameter (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_init

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    value          a list of parameter types. odd cases pad with 0x0000.
    length         length of this param containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _VALUE_STR = '!H'
    _VALUE_LEN = struct.calcsize(_VALUE_STR)

    @classmethod
    def param_type(cls):
        return PTYPE_SUPPORTED_ADDR

    def __init__(self, value=None, length=0):
        if not isinstance(value, list):
            value = [value]
        for one in value:
            assert isinstance(one, int)
        super(param_supported_addr, self).__init__(value, length)

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        value = []
        offset = cls._MIN_LEN
        while offset < length:
            one, = struct.unpack_from(cls._VALUE_STR, buf, offset)
            value.append(one)
            offset += cls._VALUE_LEN
        return cls(value, length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.param_type(), self.length))
        for one in self.value:
            buf.extend(struct.pack(param_supported_addr._VALUE_STR, one))
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)