import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def _AsciiHexDecode(input):
    """Decodes input using ASCII-Hex coding.

    Not used except to provide a test of the inverse function."""
    if not isUnicode(input):
        input = input.decode('utf-8')
    stripped = ''.join(input.split())
    assert stripped[-1] == '>', 'Invalid terminator for Ascii Hex Stream'
    stripped = stripped[:-1]
    assert len(stripped) % 2 == 0, 'Ascii Hex stream has odd number of bytes'
    return ''.join([chr(int(stripped[i:i + 2], 16)) for i in range(0, len(stripped), 2)])