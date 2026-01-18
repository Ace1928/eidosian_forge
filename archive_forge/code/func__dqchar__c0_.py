import unicodedata
def _dqchar__c0_(self):
    self._push('dqchar__c0')
    self._seq([self._bslash_, lambda: self._bind(self._esc_char_, 'c'), lambda: self._succeed(self._get('c'))])
    self._pop('dqchar__c0')