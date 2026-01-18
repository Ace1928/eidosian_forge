from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
class LF(Transform):
    """ENTER sends LF"""