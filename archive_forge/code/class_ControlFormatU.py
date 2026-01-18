import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
@llc.register_control_type
class ControlFormatU(stringify.StringifyMixin):
    """LLC sub encoder/decoder class for control U-format field.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ======================== ===============================
    Attribute                Description
    ======================== ===============================
    modifier_function1       modifier function bit
    pf_bit                   poll/final bit
    modifier_function2       modifier function bit
    ======================== ===============================
    """
    TYPE = 3
    _PACK_STR = '!B'
    _PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, modifier_function1=0, pf_bit=0, modifier_function2=0):
        super(ControlFormatU, self).__init__()
        self.modifier_function1 = modifier_function1
        self.pf_bit = pf_bit
        self.modifier_function2 = modifier_function2

    @classmethod
    def parser(cls, buf):
        assert len(buf) >= cls._PACK_LEN
        control, = struct.unpack_from(cls._PACK_STR, buf)
        assert control & 3 == cls.TYPE
        modifier_function1 = control >> 2 & 3
        pf_bit = control >> 4 & 1
        modifier_function2 = control >> 5 & 7
        return (cls(modifier_function1, pf_bit, modifier_function2), buf[cls._PACK_LEN:])

    def serialize(self):
        control = self.modifier_function2 << 5 | self.pf_bit << 4 | self.modifier_function1 << 2 | self.TYPE
        return struct.pack(self._PACK_STR, control)