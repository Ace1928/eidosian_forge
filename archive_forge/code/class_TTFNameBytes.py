from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
class TTFNameBytes(bytes):
    """class used to return named strings"""

    def __new__(cls, b, enc='utf8'):
        try:
            ustr = b.decode(enc)
        except:
            ustr = b.decode('latin1')
        self = bytes.__new__(cls, ustr.encode('utf8'))
        self.ustr = ustr
        return self