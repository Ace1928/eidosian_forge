import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def SendPing(self, data):
    return self.InternalExchange(UsbHidTransport.U2FHID_PING, data)