from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
class EOFPacketWrapper:
    """
    EOF Packet Wrapper. It uses an existing packet object, and wraps
    around it, exposing useful variables while still providing access
    to the original packet objects variables and methods.
    """

    def __init__(self, from_packet):
        if not from_packet.is_eof_packet():
            raise ValueError(f"Cannot create '{self.__class__}' object from invalid packet type")
        self.packet = from_packet
        self.warning_count, self.server_status = self.packet.read_struct('<xhh')
        if DEBUG:
            print('server_status=', self.server_status)
        self.has_next = self.server_status & SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS

    def __getattr__(self, key):
        return getattr(self.packet, key)