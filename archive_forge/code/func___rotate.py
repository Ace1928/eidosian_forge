import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def __rotate(self, s, n):
    l = len(s)
    if n < 0:
        n = l + n
    n %= l
    if not n:
        return s
    return s[-n:] + s[:l - n]