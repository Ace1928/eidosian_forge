import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(ND_NEIGHBOR_SOLICIT, ND_NEIGHBOR_ADVERT)
class nd_neighbor(_ICMPv6Payload):
    """ICMPv6 sub encoder/decoder class for Neighbor Solicitation and
    Neighbor Advertisement messages. (RFC 4861)

    This is used with os_ken.lib.packet.icmpv6.icmpv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    res            R,S,O Flags for Neighbor Advertisement.                    The 3 MSBs of "Reserved" field for Neighbor Solicitation.
    dst            Target Address
    option         a derived object of os_ken.lib.packet.icmpv6.nd_option                    or a bytearray. None if no options.
    ============== ====================
    """
    _PACK_STR = '!I16s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _ND_OPTION_TYPES = {}
    _TYPE = {'ascii': ['dst']}

    @staticmethod
    def register_nd_option_type(*args):

        def _register_nd_option_type(cls):
            nd_neighbor._ND_OPTION_TYPES[cls.option_type()] = cls
            return cls
        return _register_nd_option_type(args[0])

    def __init__(self, res=0, dst='::', option=None):
        self.res = res
        self.dst = dst
        self.option = option

    @classmethod
    def parser(cls, buf, offset):
        res, dst = struct.unpack_from(cls._PACK_STR, buf, offset)
        offset += cls._MIN_LEN
        option = None
        if len(buf) > offset:
            type_, length = struct.unpack_from('!BB', buf, offset)
            if length == 0:
                raise struct.error('Invalid length: {len}'.format(len=length))
            cls_ = cls._ND_OPTION_TYPES.get(type_)
            if cls_ is not None:
                option = cls_.parser(buf, offset)
            else:
                option = buf[offset:]
        msg = cls(res >> 29, addrconv.ipv6.bin_to_text(dst), option)
        return msg

    def serialize(self):
        res = self.res << 29
        hdr = bytearray(struct.pack(nd_neighbor._PACK_STR, res, addrconv.ipv6.text_to_bin(self.dst)))
        if self.option is not None:
            if isinstance(self.option, nd_option):
                hdr.extend(self.option.serialize())
            else:
                hdr.extend(self.option)
        return bytes(hdr)

    def __len__(self):
        length = self._MIN_LEN
        if self.option is not None:
            length += len(self.option)
        return length