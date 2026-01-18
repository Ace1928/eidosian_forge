import struct
import threading
import hid  # hidapi
import serial
from serial.serialutil import SerialBase, SerialException, PortNotOpenError, to_bytes, Timeout
def _hid_read_loop(self):
    try:
        while self.is_open:
            data = self._hid_handle.read(64, timeout_ms=100)
            if not data:
                continue
            data_len = data.pop(0)
            assert data_len == len(data)
            self._read_buffer.put(bytearray(data))
    finally:
        self._thread = None