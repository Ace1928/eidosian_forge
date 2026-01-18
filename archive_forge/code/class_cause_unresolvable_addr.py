import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_abort.register_cause_code
@chunk_error.register_cause_code
class cause_unresolvable_addr(cause_with_value):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Unresolvable Address (RFC 4960).

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
    value          Unresolvable Address. one of follows:

                   os_ken.lib.packet.sctp.param_host_addr,

                   os_ken.lib.packet.sctp.param_ipv4, or

                   os_ken.lib.packet.sctp.param_ipv6.
    length         length of this cause containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _class_prefixes = ['param_']
    _RECOGNIZED_PARAMS = {}

    @staticmethod
    def register_param_type(*args):

        def _register_param_type(cls):
            cause_unresolvable_addr._RECOGNIZED_PARAMS[cls.param_type()] = cls
            return cls
        return _register_param_type(args[0])

    @classmethod
    def cause_code(cls):
        return CCODE_UNRESOLVABLE_ADDR

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        ptype, = struct.unpack_from('!H', buf, cls._MIN_LEN)
        cls_ = cls._RECOGNIZED_PARAMS.get(ptype)
        value = cls_.parser(buf[cls._MIN_LEN:])
        return cls(value, length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.cause_code(), self.length))
        buf.extend(self.value.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)