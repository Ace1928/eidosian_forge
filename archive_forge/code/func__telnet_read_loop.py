from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _telnet_read_loop(self):
    """Read loop for the socket."""
    mode = M_NORMAL
    suboption = None
    try:
        while self.is_open:
            try:
                data = self._socket.recv(1024)
            except socket.timeout:
                continue
            except socket.error as e:
                if self.logger:
                    self.logger.debug('socket error in reader thread: {}'.format(e))
                self._read_buffer.put(None)
                break
            if not data:
                self._read_buffer.put(None)
                break
            for byte in iterbytes(data):
                if mode == M_NORMAL:
                    if byte == IAC:
                        mode = M_IAC_SEEN
                    elif suboption is not None:
                        suboption += byte
                    else:
                        self._read_buffer.put(byte)
                elif mode == M_IAC_SEEN:
                    if byte == IAC:
                        if suboption is not None:
                            suboption += IAC
                        else:
                            self._read_buffer.put(IAC)
                        mode = M_NORMAL
                    elif byte == SB:
                        suboption = bytearray()
                        mode = M_NORMAL
                    elif byte == SE:
                        self._telnet_process_subnegotiation(bytes(suboption))
                        suboption = None
                        mode = M_NORMAL
                    elif byte in (DO, DONT, WILL, WONT):
                        telnet_command = byte
                        mode = M_NEGOTIATE
                    else:
                        self._telnet_process_command(byte)
                        mode = M_NORMAL
                elif mode == M_NEGOTIATE:
                    self._telnet_negotiate_option(telnet_command, byte)
                    mode = M_NORMAL
    finally:
        if self.logger:
            self.logger.debug('read thread terminated')