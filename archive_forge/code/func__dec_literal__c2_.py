import unicodedata
def _dec_literal__c2_(self):
    self._push('dec_literal__c2')
    self._seq([lambda: self._bind(self._dec_int_lit_, 'd'), lambda: self._bind(self._exp_, 'e'), lambda: self._succeed(self._get('d') + self._get('e'))])
    self._pop('dec_literal__c2')