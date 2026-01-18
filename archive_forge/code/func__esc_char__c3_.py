import unicodedata
def _esc_char__c3_(self):
    self._seq([lambda: self._ch('r'), lambda: self._succeed('\r')])