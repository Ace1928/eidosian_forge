from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def _stop_reader(self):
    """Stop reader thread only, wait for clean exit of thread"""
    self._reader_alive = False
    if hasattr(self.serial, 'cancel_read'):
        self.serial.cancel_read()
    self.receiver_thread.join()