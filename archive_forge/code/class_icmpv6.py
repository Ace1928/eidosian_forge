import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class icmpv6(packet_base.PacketBase):
    """ICMPv6 (RFC 2463) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    type\\_         Type
    code           Code
    csum           CheckSum
                   (0 means automatically-calculate when encoding)
    data           Payload.

                   os_ken.lib.packet.icmpv6.echo object, \\
                   os_ken.lib.packet.icmpv6.nd_neighbor object, \\
                   os_ken.lib.packet.icmpv6.nd_router_solicit object, \\
                   os_ken.lib.packet.icmpv6.nd_router_advert object, \\
                   os_ken.lib.packet.icmpv6.mld object, \\
                   or a bytearray.
    ============== ====================
    """
    _PACK_STR = '!BBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _ICMPV6_TYPES = {}

    @staticmethod
    def register_icmpv6_type(*args):

        def _register_icmpv6_type(cls):
            for type_ in args:
                icmpv6._ICMPV6_TYPES[type_] = cls
            return cls
        return _register_icmpv6_type

    def __init__(self, type_=0, code=0, csum=0, data=b''):
        super(icmpv6, self).__init__()
        self.type_ = type_
        self.code = code
        self.csum = csum
        self.data = data

    @classmethod
    def parser(cls, buf):
        type_, code, csum = struct.unpack_from(cls._PACK_STR, buf)
        msg = cls(type_, code, csum)
        offset = cls._MIN_LEN
        if len(buf) > offset:
            cls_ = cls._ICMPV6_TYPES.get(type_, None)
            if cls_:
                msg.data = cls_.parser(buf, offset)
            else:
                msg.data = buf[offset:]
        return (msg, None, None)

    def serialize(self, payload, prev):
        hdr = bytearray(struct.pack(icmpv6._PACK_STR, self.type_, self.code, self.csum))
        if self.data:
            if self.type_ in icmpv6._ICMPV6_TYPES:
                assert isinstance(self.data, _ICMPv6Payload)
                hdr += self.data.serialize()
            else:
                hdr += self.data
        if self.csum == 0:
            self.csum = packet_utils.checksum_ip(prev, len(hdr), hdr + payload)
            struct.pack_into('!H', hdr, 2, self.csum)
        return hdr

    def __len__(self):
        return self._MIN_LEN + len(self.data)