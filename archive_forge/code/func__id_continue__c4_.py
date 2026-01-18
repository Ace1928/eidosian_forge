import unicodedata
def _id_continue__c4_(self):
    self._push('id_continue__c4')
    self._seq([lambda: self._bind(self._anything_, 'x'), self._id_continue__c4__s1_, lambda: self._succeed(self._get('x'))])
    self._pop('id_continue__c4')