from pyu2f import hidtransport
from pyu2f.hid import base
def Read(self):
    """Read from the device.

    Reads from the fake hid device.  This function only works if a transaction
    is open and a complete write has taken place.  If so, it will return the
    next reply packet.  It should be called repeatedly until all expected
    data has been received.  It always reads one report.

    Returns:
      A list of ints containing the next packet.

    Raises:
      UnsupportedCommandError: if the requested amount is not 64.
    """
    if not self.transaction_active or not self.full_packet_received:
        return None
    ret = None
    if self.busy_count > 0:
        ret = hidtransport.UsbHidTransport.InitPacket(64, self.init_packet.cid, hidtransport.UsbHidTransport.U2FHID_ERROR, 1, hidtransport.UsbHidTransport.ERR_CHANNEL_BUSY)
        self.busy_count -= 1
    elif self.reply:
        next_frame = self.reply[0:59]
        self.reply = self.reply[59:]
        ret = hidtransport.UsbHidTransport.ContPacket(64, self.init_packet.cid, self.seq, next_frame)
        self.seq += 1
    else:
        self.InternalGenerateReply()
        first_frame = self.reply[0:57]
        ret = hidtransport.UsbHidTransport.InitPacket(64, self.init_packet.cid, self.init_packet.cmd, len(self.reply), first_frame)
        self.reply = self.reply[57:]
    if not self.reply:
        self.reply = None
        self.transaction_active = False
        self.seq = 0
    return ret.ToWireFormat()