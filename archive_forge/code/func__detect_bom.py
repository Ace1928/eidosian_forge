from __future__ import unicode_literals
import codecs
from .labels import LABELS
def _detect_bom(input):
    """Return (bom_encoding, input), with any BOM removed from the input."""
    if input.startswith(b'\xff\xfe'):
        return (_UTF16LE, input[2:])
    if input.startswith(b'\xfe\xff'):
        return (_UTF16BE, input[2:])
    if input.startswith(b'\xef\xbb\xbf'):
        return (UTF8, input[3:])
    return (None, input)