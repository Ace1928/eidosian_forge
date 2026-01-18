from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def _assignState(self, doc, asciiReadable=None, namePrefix=None):
    """convenience function for those wishing to roll their own state properties"""
    if asciiReadable is None:
        asciiReadable = self._asciiReadable
    try:
        state = self.state[doc]
    except KeyError:
        state = self.state[doc] = TTFont.State(asciiReadable, self)
        if namePrefix is not None:
            state.namePrefix = namePrefix
    return state