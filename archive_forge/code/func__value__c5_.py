import unicodedata
def _value__c5_(self):
    self._push('value__c5')
    self._seq([lambda: self._bind(self._string_, 'v'), lambda: self._succeed(['string', self._get('v')])])
    self._pop('value__c5')