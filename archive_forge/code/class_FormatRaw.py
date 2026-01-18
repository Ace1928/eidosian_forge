from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
class FormatRaw(object):
    """Forward only RX and TX data to output."""

    def __init__(self, output, color):
        self.output = output
        self.color = color
        self.rx_color = '\x1b[32m'
        self.tx_color = '\x1b[31m'

    def rx(self, data):
        """show received data"""
        if self.color:
            self.output.write(self.rx_color)
        self.output.write(data)
        self.output.flush()

    def tx(self, data):
        """show transmitted data"""
        if self.color:
            self.output.write(self.tx_color)
        self.output.write(data)
        self.output.flush()

    def control(self, name, value):
        """(do not) show control calls"""
        pass