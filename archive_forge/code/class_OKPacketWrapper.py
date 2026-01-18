from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
class OKPacketWrapper:
    """
    OK Packet Wrapper. It uses an existing packet object, and wraps
    around it, exposing useful variables while still providing access
    to the original packet objects variables and methods.
    """

    def __init__(self, from_packet):
        if not from_packet.is_ok_packet():
            raise ValueError('Cannot create ' + str(self.__class__.__name__) + ' object from invalid packet type')
        self.packet = from_packet
        self.packet.advance(1)
        self.affected_rows = self.packet.read_length_encoded_integer()
        self.insert_id = self.packet.read_length_encoded_integer()
        self.server_status, self.warning_count = self.read_struct('<HH')
        self.message = self.packet.read_all()
        self.has_next = self.server_status & SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS

    def __getattr__(self, key):
        return getattr(self.packet, key)