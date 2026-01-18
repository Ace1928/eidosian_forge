import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def SendMsgBytes(self, msg):
    r = self.InternalExchange(UsbHidTransport.U2FHID_MSG, msg)
    return r