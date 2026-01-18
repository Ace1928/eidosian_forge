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
class svlan(_vlan):
    """S-VLAN (IEEE 802.1ad) header encoder/decoder class.


    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== ====================
    Attribute      Description
    ============== ====================
    pcp            Priority Code Point
    cfi            Canonical Format Indicator.
                   In a case to be used as B-TAG,
                   this field means DEI(Drop Eligible Indication).
    vid            VLAN Identifier
    ethertype      EtherType
    ============== ====================
    """

    def __init__(self, pcp=0, cfi=0, vid=0, ethertype=ether.ETH_TYPE_8021Q):
        super(svlan, self).__init__(pcp, cfi, vid, ethertype)

    @classmethod
    def get_packet_type(cls, type_):
        return cls._TYPES.get(type_)