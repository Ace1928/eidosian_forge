import unicodedata
def _num_literal__c0_(self):
    self._push('num_literal__c0')
    self._seq([lambda: self._ch('-'), lambda: self._bind(self._num_literal_, 'n'), lambda: self._succeed('-' + self._get('n'))])
    self._pop('num_literal__c0')