from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _telnet_negotiate_option(self, command, option):
    """Process incoming DO, DONT, WILL, WONT."""
    known = False
    for item in self._telnet_options:
        if item.option == option:
            item.process_incoming(command)
            known = True
    if not known:
        if command == WILL or command == DO:
            self.telnet_send_option(DONT if command == WILL else WONT, option)
            if self.logger:
                self.logger.warning('rejected Telnet option: {!r}'.format(option))