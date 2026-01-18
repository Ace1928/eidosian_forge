from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
def _addCSD(self, s, m):
    i = c = 0
    v = 1
    V = self.values
    while i < len(s):
        c += v * V[s[-(i + 1)]]
        i += 1
        v += 1
        if v == m:
            v = 1
    return s + self.chars[c % 11]