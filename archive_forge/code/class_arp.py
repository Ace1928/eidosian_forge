import struct
from os_ken.lib import addrconv
from . import ether_types as ether
from . import packet_base
class arp(packet_base.PacketBase):
    """ARP (RFC 826) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    IPv4 addresses are represented as a string like '192.0.2.1'.
    MAC addresses are represented as a string like '08:60:6e:7f:74:e7'.
    __init__ takes the corresponding args in this order.

    ============== ===================================== =====================
    Attribute      Description                           Example
    ============== ===================================== =====================
    hwtype         Hardware address.
    proto          Protocol address.
    hlen           byte length of each hardware address.
    plen           byte length of each protocol address.
    opcode         operation codes.
    src_mac        Hardware address of sender.           '08:60:6e:7f:74:e7'
    src_ip         Protocol address of sender.           '192.0.2.1'
    dst_mac        Hardware address of target.           '00:00:00:00:00:00'
    dst_ip         Protocol address of target.           '192.0.2.2'
    ============== ===================================== =====================
    """
    _PACK_STR = '!HHBBH6s4s6s4s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['src_mac', 'src_ip', 'dst_mac', 'dst_ip']}

    def __init__(self, hwtype=ARP_HW_TYPE_ETHERNET, proto=ether.ETH_TYPE_IP, hlen=6, plen=4, opcode=ARP_REQUEST, src_mac='ff:ff:ff:ff:ff:ff', src_ip='0.0.0.0', dst_mac='ff:ff:ff:ff:ff:ff', dst_ip='0.0.0.0'):
        super(arp, self).__init__()
        self.hwtype = hwtype
        self.proto = proto
        self.hlen = hlen
        self.plen = plen
        self.opcode = opcode
        self.src_mac = src_mac
        self.src_ip = src_ip
        self.dst_mac = dst_mac
        self.dst_ip = dst_ip

    @classmethod
    def parser(cls, buf):
        hwtype, proto, hlen, plen, opcode, src_mac, src_ip, dst_mac, dst_ip = struct.unpack_from(cls._PACK_STR, buf)
        return (cls(hwtype, proto, hlen, plen, opcode, addrconv.mac.bin_to_text(src_mac), addrconv.ipv4.bin_to_text(src_ip), addrconv.mac.bin_to_text(dst_mac), addrconv.ipv4.bin_to_text(dst_ip)), None, buf[arp._MIN_LEN:])

    def serialize(self, payload, prev):
        return struct.pack(arp._PACK_STR, self.hwtype, self.proto, self.hlen, self.plen, self.opcode, addrconv.mac.text_to_bin(self.src_mac), addrconv.ipv4.text_to_bin(self.src_ip), addrconv.mac.text_to_bin(self.dst_mac), addrconv.ipv4.text_to_bin(self.dst_ip))