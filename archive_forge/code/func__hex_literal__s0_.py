import unicodedata
def _hex_literal__s0_(self):
    self._choose([lambda: self._str('0x'), lambda: self._str('0X')])