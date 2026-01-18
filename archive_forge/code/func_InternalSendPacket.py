import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalSendPacket(self, packet):
    wire = packet.ToWireFormat()
    self.logger.debug('sending packet: ' + str(wire))
    self.hid_device.Write(wire)