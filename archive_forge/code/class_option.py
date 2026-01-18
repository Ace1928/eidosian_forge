import random
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import stringify
from . import packet_base
class option(stringify.StringifyMixin):
    """DHCP (RFC 2132) options encoder/decoder class.

    This is used with os_ken.lib.packet.dhcp.dhcp.options.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    tag            Option type.                   (except for the 'magic cookie', 'pad option'                    and 'end option'.)
    value          Option's value.                   (set the value that has been converted to hexadecimal.)
    length         Option's value length.                   (calculated automatically from the length of value.)
    ============== ====================
    """
    _UNPACK_STR = '!B'
    _MIN_LEN = struct.calcsize(_UNPACK_STR)

    def __init__(self, tag, value, length=0):
        super(option, self).__init__()
        self.tag = tag
        self.value = value
        self.length = length

    @classmethod
    def parser(cls, buf):
        tag = struct.unpack_from(cls._UNPACK_STR, buf)[0]
        if tag == DHCP_END_OPT or tag == DHCP_PAD_OPT:
            return None
        buf = buf[cls._MIN_LEN:]
        length = struct.unpack_from(cls._UNPACK_STR, buf)[0]
        buf = buf[cls._MIN_LEN:]
        value_unpack_str = '%ds' % length
        value = struct.unpack_from(value_unpack_str, buf)[0]
        return cls(tag, value, length)

    def serialize(self):
        self.length = len(self.value)
        options_pack_str = '!BB%ds' % self.length
        return struct.pack(options_pack_str, self.tag, self.length, self.value)