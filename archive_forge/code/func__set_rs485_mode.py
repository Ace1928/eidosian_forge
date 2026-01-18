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
def _set_rs485_mode(self, rs485_settings):
    buf = array.array('i', [0] * 8)
    try:
        fcntl.ioctl(self.fd, TIOCGRS485, buf)
        buf[0] |= SER_RS485_ENABLED
        if rs485_settings is not None:
            if rs485_settings.loopback:
                buf[0] |= SER_RS485_RX_DURING_TX
            else:
                buf[0] &= ~SER_RS485_RX_DURING_TX
            if rs485_settings.rts_level_for_tx:
                buf[0] |= SER_RS485_RTS_ON_SEND
            else:
                buf[0] &= ~SER_RS485_RTS_ON_SEND
            if rs485_settings.rts_level_for_rx:
                buf[0] |= SER_RS485_RTS_AFTER_SEND
            else:
                buf[0] &= ~SER_RS485_RTS_AFTER_SEND
            if rs485_settings.delay_before_tx is not None:
                buf[1] = int(rs485_settings.delay_before_tx * 1000)
            if rs485_settings.delay_before_rx is not None:
                buf[2] = int(rs485_settings.delay_before_rx * 1000)
        else:
            buf[0] = 0
        fcntl.ioctl(self.fd, TIOCSRS485, buf)
    except IOError as e:
        raise ValueError('Failed to set RS485 mode: {}'.format(e))