import unicodedata
def _object__c0_(self):
    self._push('object__c0')
    self._seq([lambda: self._ch('{'), self._sp_, lambda: self._bind(self._member_list_, 'v'), self._sp_, lambda: self._ch('}'), lambda: self._succeed(self._get('v'))])
    self._pop('object__c0')