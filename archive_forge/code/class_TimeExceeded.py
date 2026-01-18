import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
@icmp.register_icmp_type(ICMP_TIME_EXCEEDED)
class TimeExceeded(_ICMPv4Payload):
    """ICMP sub encoder/decoder class for Time Exceeded Message.

    This is used with os_ken.lib.packet.icmp.icmp for
    ICMP Time Exceeded Message.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    [RFC4884] introduced 8-bit data length attribute.

    .. tabularcolumns:: |l|L|

    ============== ====================
    Attribute      Description
    ============== ====================
    data_len       data length
    data           Internet Header + leading octets of original datagram
    ============== ====================
    """
    _PACK_STR = '!xBxx'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, data_len=0, data=None):
        if data_len >= 0 and data_len <= 255:
            self.data_len = data_len
        else:
            raise ValueError('Specified data length (%d) is invalid.' % data_len)
        self.data = data

    @classmethod
    def parser(cls, buf, offset):
        data_len, = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(data_len)
        offset += cls._MIN_LEN
        if len(buf) > offset:
            msg.data = buf[offset:]
        return msg

    def serialize(self):
        hdr = bytearray(struct.pack(TimeExceeded._PACK_STR, self.data_len))
        if self.data is not None:
            hdr += self.data
        return hdr

    def __len__(self):
        length = self._MIN_LEN
        if self.data is not None:
            length += len(self.data)
        return length