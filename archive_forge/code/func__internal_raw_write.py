from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _internal_raw_write(self, data):
    """internal socket write with no data escaping. used to send telnet stuff."""
    with self._write_lock:
        self._socket.sendall(data)