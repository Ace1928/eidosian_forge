import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def _chunker(src, dst=[], chunkSize=60):
    for i in range(0, len(src), chunkSize):
        dst.append(src[i:i + chunkSize])
    return dst