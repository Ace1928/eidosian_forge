import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
class ContPacket(object):
    """Represents a continutation U2FHID packet.

    Represents a continutation U2FHID packet.  These packets follow
    the intial packet and contains the remaining data in a particular
    message.

    Attributes:
      packet_size: The size of the hid report (packet) used.  Usually 64.
      cid: The channel id for the connection to the device.
      seq: The sequence number for this continuation packet.  The first
          continuation packet is 0 and it increases from there.
      payload:  The payload to put into this continuation packet.  This
          must be less than packet_size - 5 (the overhead of the
          continuation packet is 5).
    """

    def __init__(self, packet_size, cid, seq, payload):
        self.packet_size = packet_size
        self.cid = cid
        self.seq = seq
        self.payload = payload
        if len(payload) > self.packet_size - 5:
            raise errors.InvalidPacketError()
        if seq > 127:
            raise errors.InvalidPacketError()

    def ToWireFormat(self):
        """Serializes the packet."""
        ret = bytearray(self.packet_size)
        ret[0:4] = self.cid
        ret[4] = self.seq
        ret[5:5 + len(self.payload)] = self.payload
        return list(map(int, ret))

    @staticmethod
    def FromWireFormat(packet_size, data):
        """Derializes the packet.

      Deserializes the packet from wire format.

      Args:
        packet_size: The size of all packets (usually 64)
        data: List of ints or bytearray containing the data from the wire.

      Returns:
        InitPacket object for specified data

      Raises:
        InvalidPacketError: if the data isn't a valid ContPacket
      """
        ba = bytearray(data)
        if len(ba) != packet_size:
            raise errors.InvalidPacketError()
        cid = ba[0:4]
        seq = ba[4]
        payload = ba[5:]
        return UsbHidTransport.ContPacket(packet_size, cid, seq, payload)