from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
class NoControls(NoTerminal):
    """Remove all control codes, incl. CR+LF"""
    REPLACEMENT_MAP = dict(((x, 9216 + x) for x in range(32)))
    REPLACEMENT_MAP.update({32: 9251, 127: 9249, 155: 9253})