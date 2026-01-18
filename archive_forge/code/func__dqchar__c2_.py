import unicodedata
def _dqchar__c2_(self):
    self._push('dqchar__c2')
    self._seq([lambda: self._not(self._bslash_), lambda: self._not(self._dquote_), lambda: self._not(self._eol_), lambda: self._bind(self._anything_, 'c'), lambda: self._succeed(self._get('c'))])
    self._pop('dqchar__c2')