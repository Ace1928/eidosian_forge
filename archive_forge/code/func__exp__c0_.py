import unicodedata
def _exp__c0_(self):
    self._push('exp__c0')
    self._seq([self._exp__c0__s0_, lambda: self._bind(self._exp__c0__s1_l_, 's'), self._exp__c0__s2_, lambda: self._succeed('e' + self._get('s') + self._join('', self._get('ds')))])
    self._pop('exp__c0')