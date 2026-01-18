import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
@llc.register_control_type
class ControlFormatS(stringify.StringifyMixin):
    """LLC sub encoder/decoder class for control S-format field.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ======================== ===============================
    Attribute                Description
    ======================== ===============================
    supervisory_function     supervisory function bit
    pf_bit                   poll/final bit
    receive_sequence_number  sender receive sequence number
    ======================== ===============================
    """
    TYPE = 1
    _PACK_STR = '!H'
    _PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, supervisory_function=0, pf_bit=0, receive_sequence_number=0):
        super(ControlFormatS, self).__init__()
        self.supervisory_function = supervisory_function
        self.pf_bit = pf_bit
        self.receive_sequence_number = receive_sequence_number

    @classmethod
    def parser(cls, buf):
        assert len(buf) >= cls._PACK_LEN
        control, = struct.unpack_from(cls._PACK_STR, buf)
        assert control >> 8 & 3 == cls.TYPE
        supervisory_function = control >> 10 & 3
        pf_bit = control >> 8 & 1
        receive_sequence_number = control >> 1 & 127
        return (cls(supervisory_function, pf_bit, receive_sequence_number), buf[cls._PACK_LEN:])

    def serialize(self):
        control = self.supervisory_function << 10 | self.TYPE << 8 | self.receive_sequence_number << 1 | self.pf_bit
        return struct.pack(self._PACK_STR, control)