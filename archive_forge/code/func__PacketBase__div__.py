import inspect
import struct
import base64
from . import packet_base
from . import ethernet
from os_ken import utils
from os_ken.lib.stringify import StringifyMixin
def _PacketBase__div__(self, trailer):
    pkt = Packet()
    pkt.add_protocol(self)
    pkt.add_protocol(trailer)
    return pkt