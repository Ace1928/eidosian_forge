import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalInit(self):
    """Initializes the device and obtains channel id."""
    self.cid = UsbHidTransport.U2FHID_BROADCAST_CID
    nonce = bytearray(os.urandom(8))
    r = self.InternalExchange(UsbHidTransport.U2FHID_INIT, nonce)
    if len(r) < 17:
        raise errors.HidError('unexpected init reply len')
    if r[0:8] != nonce:
        raise errors.HidError('nonce mismatch')
    self.cid = bytearray(r[8:12])
    self.u2fhid_version = r[12]