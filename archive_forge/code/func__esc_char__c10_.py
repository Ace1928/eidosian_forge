import unicodedata
def _esc_char__c10_(self):
    self._seq([lambda: self._ch('0'), lambda: self._not(self._digit_), lambda: self._succeed('\x00')])