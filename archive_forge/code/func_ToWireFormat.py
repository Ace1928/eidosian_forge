import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def ToWireFormat(self):
    """Serializes the packet."""
    ret = bytearray(self.packet_size)
    ret[0:4] = self.cid
    ret[4] = self.seq
    ret[5:5 + len(self.payload)] = self.payload
    return list(map(int, ret))