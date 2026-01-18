import unicodedata
def _dec_literal__c4_(self):
    self._push('dec_literal__c4')
    self._seq([lambda: self._bind(self._frac_, 'f'), lambda: self._bind(self._exp_, 'e'), lambda: self._succeed(self._get('f') + self._get('e'))])
    self._pop('dec_literal__c4')