import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_init.register_param_type
@chunk_init_ack.register_param_type
@cause_unresolvable_addr.register_param_type
@cause_restart_with_new_addr.register_param_type
class param_ipv4(param):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for IPv4 Address Parameter (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_init
    - os_ken.lib.packet.sctp.chunk_init_ack

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    value          IPv4 address of the sending endpoint.
    length         length of this param containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """
    _TYPE = {'ascii': ['value']}

    @classmethod
    def param_type(cls):
        return PTYPE_IPV4

    def __init__(self, value='127.0.0.1', length=0):
        super(param_ipv4, self).__init__(value, length)

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        value = None
        if cls._MIN_LEN < length:
            fmt = '%ds' % (length - cls._MIN_LEN)
            value, = struct.unpack_from(fmt, buf, cls._MIN_LEN)
        return cls(addrconv.ipv4.bin_to_text(value), length)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.param_type(), self.length))
        if self.value:
            buf.extend(addrconv.ipv4.text_to_bin(self.value))
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)