import struct
from . import packet_base
from . import vlan
from . import mpls
from . import ether_types as ether
from os_ken.lib import addrconv
class linuxcooked(packet_base.PacketBase):
    _PACK_STR = '!HHH8sH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, pkt_type, arphrd_type, address_length, address, proto_type):
        super(linuxcooked, self).__init__()
        self.pkt_type = pkt_type
        self.arphrd_type = arphrd_type
        self.address_length = address_length
        self.address = address
        self.proto_type = proto_type

    @classmethod
    def parser(cls, buf):
        pkt_type, arphrd_type, address_length, addres, proto_type = struct.unpack_from(cls._PACK_STR, buf)
        l = cls(pkt_type, arphrd_type, address_length, addres, proto_type)
        return (l, linuxcooked.get_packet_type(proto_type), buf[linuxcooked._MIN_LEN:])

    @classmethod
    def get_packet_type(cls, type_):
        """Override method for the ethernet IEEE802.3 Length/Type
        field (self.ethertype).

        If the value of Length/Type field is less than or equal to
        1500 decimal(05DC hexadecimal), it means Length interpretation
        and be passed to the LLC sublayer."""
        if type_ <= ether.ETH_TYPE_IEEE802_3:
            type_ = ether.ETH_TYPE_IEEE802_3
        return cls._TYPES.get(type_)