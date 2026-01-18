import unicodedata
def _grammar_(self):
    self._push('grammar')
    self._seq([self._sp_, lambda: self._bind(self._value_, 'v'), self._sp_, self._end_, lambda: self._succeed(self._get('v'))])
    self._pop('grammar')