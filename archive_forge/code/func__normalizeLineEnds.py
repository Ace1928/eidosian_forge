import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def _normalizeLineEnds(text, desired='\r\n', unlikely='\x00\x01\x02\x03'):
    """Normalizes different line end character(s).

    Ensures all instances of CR, LF and CRLF end up as
    the specified one."""
    return text.replace('\r\n', unlikely).replace('\r', unlikely).replace('\n', unlikely).replace(unlikely, desired)