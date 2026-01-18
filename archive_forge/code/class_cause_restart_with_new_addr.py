import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_abort.register_cause_code
@chunk_error.register_cause_code
class cause_restart_with_new_addr(cause_with_value):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Restart of an Association with New
    Addresses (RFC 4960).

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
    value          New Address TLVs.
    length         length of this cause containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _class_prefixes = ['param_']
    _RECOGNIZED_PARAMS = {}

    @staticmethod
    def register_param_type(*args):

        def _register_param_type(cls):
            cause_restart_with_new_addr._RECOGNIZED_PARAMS[cls.param_type()] = cls
            return cls
        return _register_param_type(args[0])

    @classmethod
    def cause_code(cls):
        return CCODE_RESTART_WITH_NEW_ADDR

    def __init__(self, value=None, length=0):
        if not isinstance(value, list):
            value = [value]
        super(cause_restart_with_new_addr, self).__init__(value, length)

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        value = []
        offset = cls._MIN_LEN
        while offset < length:
            ptype, = struct.unpack_from('!H', buf, offset)
            cls_ = cls._RECOGNIZED_PARAMS.get(ptype)
            if not cls_:
                break
            ins = cls_.parser(buf[offset:])
            value.append(ins)
            offset += len(ins)
        return cls(value, length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.cause_code(), self.length))
        for one in self.value:
            buf.extend(one.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)