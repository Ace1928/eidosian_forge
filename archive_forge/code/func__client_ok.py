from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _client_ok(self):
    """        callback of telnet option. It gets called when option is activated.
        This one here is used to detect when the client agrees on RFC 2217. A
        flag is set so that other functions like check_modem_lines know if the
        client is OK.
        """
    self._client_is_rfc2217 = True
    if self.logger:
        self.logger.info('client accepts RFC 2217')
    self.check_modem_lines(force_notification=True)