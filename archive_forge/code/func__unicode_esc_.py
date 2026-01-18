import unicodedata
def _unicode_esc_(self):
    self._push('unicode_esc')
    self._seq([lambda: self._ch('u'), lambda: self._bind(self._hex_, 'a'), lambda: self._bind(self._hex_, 'b'), lambda: self._bind(self._hex_, 'c'), lambda: self._bind(self._hex_, 'd'), lambda: self._succeed(self._xtou(self._get('a') + self._get('b') + self._get('c') + self._get('d')))])
    self._pop('unicode_esc')