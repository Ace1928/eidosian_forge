import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def _encode_timer(timer):
    return int(timer) * 256