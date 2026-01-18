import unicodedata
def _frac_(self):
    self._push('frac')
    self._seq([lambda: self._ch('.'), self._frac__s1_, lambda: self._succeed('.' + self._join('', self._get('ds')))])
    self._pop('frac')