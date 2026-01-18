import struct
import logging
from . import packet_base
from os_ken.lib import type_desc
class vxlan(packet_base.PacketBase):
    """VXLAN (RFC 7348) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    vni            VXLAN Network Identifier
    ============== ====================
    """
    _PACK_STR = '!II'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, vni):
        super(vxlan, self).__init__()
        self.vni = vni

    @classmethod
    def parser(cls, buf):
        flags_reserved, vni_rserved = struct.unpack_from(cls._PACK_STR, buf)
        assert 1 << 3 == flags_reserved >> 24
        from os_ken.lib.packet import ethernet
        return (cls(vni_rserved >> 8), ethernet.ethernet, buf[cls._MIN_LEN:])

    def serialize(self, payload, prev):
        return struct.pack(self._PACK_STR, 1 << 3 + 24, self.vni << 8)