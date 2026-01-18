import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@nd_router_advert.register_nd_option_type
class nd_option_pi(nd_option):
    """ICMPv6 sub encoder/decoder class for Neighbor discovery
    Prefix Information Option. (RFC 4861)

    This is used with os_ken.lib.packet.icmpv6.nd_router_advert.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    length         length of the option. \\
                   (0 means automatically-calculate when encoding)
    pl             Prefix Length.
    res1           L,A,R\\* Flags for Prefix Information.
    val_l          Valid Lifetime.
    pre_l          Preferred Lifetime.
    res2           This field is unused. It MUST be initialized to zero.
    prefix         An IP address or a prefix of an IP address.
    ============== ====================

    \\*R flag is defined in (RFC 3775)
    """
    _PACK_STR = '!BBBBIII16s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['prefix']}

    @classmethod
    def option_type(cls):
        return ND_OPTION_PI

    def __init__(self, length=0, pl=0, res1=0, val_l=0, pre_l=0, res2=0, prefix='::'):
        super(nd_option_pi, self).__init__(self.option_type(), length)
        self.pl = pl
        self.res1 = res1
        self.val_l = val_l
        self.pre_l = pre_l
        self.res2 = res2
        self.prefix = prefix

    @classmethod
    def parser(cls, buf, offset):
        _, length, pl, res1, val_l, pre_l, res2, prefix = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(length, pl, res1 >> 5, val_l, pre_l, res2, addrconv.ipv6.bin_to_text(prefix))
        return msg

    def serialize(self):
        res1 = self.res1 << 5
        hdr = bytearray(struct.pack(self._PACK_STR, self.option_type(), self.length, self.pl, res1, self.val_l, self.pre_l, self.res2, addrconv.ipv6.text_to_bin(self.prefix)))
        if 0 == self.length:
            self.length = len(hdr) // 8
            struct.pack_into('!B', hdr, 1, self.length)
        return bytes(hdr)