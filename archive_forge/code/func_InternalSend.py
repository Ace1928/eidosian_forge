import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalSend(self, cmd, payload):
    """Sends a message to the device, including fragmenting it."""
    length_to_send = len(payload)
    max_payload = self.packet_size - 7
    first_frame = payload[0:max_payload]
    first_packet = UsbHidTransport.InitPacket(self.packet_size, self.cid, cmd, len(payload), first_frame)
    del payload[0:max_payload]
    length_to_send -= len(first_frame)
    self.InternalSendPacket(first_packet)
    seq = 0
    while length_to_send > 0:
        max_payload = self.packet_size - 5
        next_frame = payload[0:max_payload]
        del payload[0:max_payload]
        length_to_send -= len(next_frame)
        next_packet = UsbHidTransport.ContPacket(self.packet_size, self.cid, seq, next_frame)
        self.InternalSendPacket(next_packet)
        seq += 1