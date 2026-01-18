from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _reset_input_buffer(self):
    """Clear input buffer, discarding all that is in the buffer."""
    termios.tcflush(self.fd, termios.TCIFLUSH)