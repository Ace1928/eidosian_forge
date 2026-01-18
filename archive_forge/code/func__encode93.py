from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
def _encode93(str):
    s = list(str)
    s.reverse()
    i = 0
    v = 1
    c = 0
    while i < len(s):
        c = c + v * _patterns[s[i]][1]
        i = i + 1
        v = v + 1
        if v > 20:
            v = 1
    s.insert(0, _charsbyval[c % 47])
    i = 0
    v = 1
    c = 0
    while i < len(s):
        c = c + v * _patterns[s[i]][1]
        i = i + 1
        v = v + 1
        if v > 15:
            v = 1
    s.insert(0, _charsbyval[c % 47])
    s.reverse()
    return ''.join(s)