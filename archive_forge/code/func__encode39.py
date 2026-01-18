from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import Barcode
from string import ascii_uppercase, ascii_lowercase, digits as string_digits
def _encode39(value, cksum, stop):
    v = sum([_patterns[c][1] for c in value]) % 43
    if cksum:
        value += _stdchrs[v]
    if stop:
        value = '*' + value + '*'
    return value