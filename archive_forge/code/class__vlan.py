import abc
import struct
from . import packet_base
from . import arp
from . import ipv4
from . import ipv6
from . import lldp
from . import slow
from . import llc
from . import pbb
from . import cfm
from . import ether_types as ether
class _vlan(packet_base.PacketBase, metaclass=abc.ABCMeta):
    _PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @abc.abstractmethod
    def __init__(self, pcp, cfi, vid, ethertype):
        super(_vlan, self).__init__()
        self.pcp = pcp
        self.cfi = cfi
        self.vid = vid
        self.ethertype = ethertype

    @classmethod
    def parser(cls, buf):
        tci, ethertype = struct.unpack_from(cls._PACK_STR, buf)
        pcp = tci >> 13
        cfi = tci >> 12 & 1
        vid = tci & (1 << 12) - 1
        return (cls(pcp, cfi, vid, ethertype), vlan.get_packet_type(ethertype), buf[vlan._MIN_LEN:])

    def serialize(self, payload, prev):
        tci = self.pcp << 13 | self.cfi << 12 | self.vid
        return struct.pack(vlan._PACK_STR, tci, self.ethertype)