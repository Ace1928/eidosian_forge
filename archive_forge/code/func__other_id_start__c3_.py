import unicodedata
def _other_id_start__c3_(self):
    self._push('other_id_start__c3')
    self._seq([lambda: self._bind(self._anything_, 'x'), self._other_id_start__c3__s1_, lambda: self._succeed(self._get('x'))])
    self._pop('other_id_start__c3')