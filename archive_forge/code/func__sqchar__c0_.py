import unicodedata
def _sqchar__c0_(self):
    self._push('sqchar__c0')
    self._seq([self._bslash_, lambda: self._bind(self._esc_char_, 'c'), lambda: self._succeed(self._get('c'))])
    self._pop('sqchar__c0')