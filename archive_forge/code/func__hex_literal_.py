import unicodedata
def _hex_literal_(self):
    self._push('hex_literal')
    self._seq([self._hex_literal__s0_, self._hex_literal__s1_, lambda: self._succeed('0x' + self._join('', self._get('hs')))])
    self._pop('hex_literal')