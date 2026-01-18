import struct
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import ether_types
class geneve(packet_base.PacketBase):
    """Geneve (RFC draft-ietf-nvo3-geneve-03) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ========================================================
    Attribute      Description
    ============== ========================================================
    version        Version.
    opt_len        The length of the options fields.
    flags          Flag field for OAM packet and Critical options present.
    protocol       Protocol Type field.
                   The Protocol Type is defined as "ETHER TYPES".
    vni            Identifier for unique element of virtual network.
    options        List of ``Option*`` instance.
    ============== ========================================================
    """
    _HEADER_FMT = '!BBHI'
    _MIN_LEN = struct.calcsize(_HEADER_FMT)
    OAM_PACKET_FLAG = 1 << 7
    CRITICAL_OPTIONS_FLAG = 1 << 6

    def __init__(self, version=0, opt_len=0, flags=0, protocol=ether_types.ETH_TYPE_TEB, vni=None, options=None):
        super(geneve, self).__init__()
        self.version = version
        self.opt_len = opt_len
        assert flags & 63 == 0
        self.flags = flags
        self.protocol = protocol
        self.vni = vni
        for o in options:
            assert isinstance(o, Option)
        self.options = options

    @classmethod
    def parser(cls, buf):
        ver_opt_len, flags, protocol, vni = struct.unpack_from(cls._HEADER_FMT, buf)
        version = ver_opt_len >> 6
        opt_len = (ver_opt_len & 63) * 4
        opt_bin = buf[cls._MIN_LEN:cls._MIN_LEN + opt_len]
        options = []
        while opt_bin:
            option, opt_bin = Option.parser(opt_bin)
            options.append(option)
        msg = cls(version, opt_len, flags, protocol, vni >> 8, options)
        from . import ethernet
        geneve._TYPES = ethernet.ethernet._TYPES
        geneve.register_packet_type(ethernet.ethernet, ether_types.ETH_TYPE_TEB)
        return (msg, geneve.get_packet_type(protocol), buf[cls._MIN_LEN + opt_len:])

    def serialize(self, payload=None, prev=None):
        tunnel_options = bytearray()
        for o in self.options:
            tunnel_options += o.serialize()
        self.opt_len = len(tunnel_options)
        opt_len = self.opt_len // 4
        return struct.pack(self._HEADER_FMT, self.version << 6 | opt_len, self.flags, self.protocol, self.vni << 8) + tunnel_options