import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def SendWink(self):
    return self.InternalExchange(UsbHidTransport.U2FHID_WINK, bytearray([]))