import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@bpdu.register_bpdu_type
class TopologyChangeNotificationBPDUs(bpdu):
    """Topology Change Notification BPDUs(IEEE 802.1D)
    header encoder/decoder class.
    """
    VERSION_ID = PROTOCOLVERSION_ID_BPDU
    BPDU_TYPE = TYPE_TOPOLOGY_CHANGE_BPDU
    _PACK_STR = ''
    PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self):
        super(TopologyChangeNotificationBPDUs, self).__init__()

    @classmethod
    def parser(cls, buf):
        return (cls(), None, buf[bpdu._PACK_LEN:])