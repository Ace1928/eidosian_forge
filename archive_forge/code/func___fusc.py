import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def __fusc(self, s):
    slen = len(s)
    return list(map(lambda x, y: x ^ y, s, list(map(ord, ((int(slen / self._klen) + 1) * self._k)[:slen]))))