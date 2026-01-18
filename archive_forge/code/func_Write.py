from pyu2f import hidtransport
from pyu2f.hid import base
def Write(self, data):
    """Write to the device.

    Writes to the fake hid device.  This function is stateful: if a transaction
    is currently open with the client, it will continue to accumulate data
    for the client->device messages until the expected size is reached.

    Args:
      data: A list of integers to accept as data payload.  It should be 64 bytes
          in length: just the report data--NO report ID.
    """
    if len(data) < 64:
        data = bytearray(data) + bytearray((0 for i in range(0, 64 - len(data))))
    if not self.transaction_active:
        self.transaction_active = True
        self.init_packet = hidtransport.UsbHidTransport.InitPacket.FromWireFormat(64, data)
        self.packet_body = self.init_packet.payload
        self.full_packet_received = False
        self.received_packets.append(self.init_packet)
    else:
        cont_packet = hidtransport.UsbHidTransport.ContPacket.FromWireFormat(64, data)
        self.packet_body += cont_packet.payload
        self.received_packets.append(cont_packet)
    if len(self.packet_body) >= self.init_packet.size:
        self.packet_body = self.packet_body[0:self.init_packet.size]
        self.full_packet_received = True