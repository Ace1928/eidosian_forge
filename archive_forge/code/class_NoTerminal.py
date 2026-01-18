from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
class NoTerminal(Transform):
    """remove typical terminal control codes from input"""
    REPLACEMENT_MAP = dict(((x, 9216 + x) for x in range(32) if unichr(x) not in '\r\n\x08\t'))
    REPLACEMENT_MAP.update({127: 9249, 155: 9253})

    def rx(self, text):
        return text.translate(self.REPLACEMENT_MAP)
    echo = rx