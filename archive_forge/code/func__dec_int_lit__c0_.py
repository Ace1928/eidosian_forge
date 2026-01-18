import unicodedata
def _dec_int_lit__c0_(self):
    self._seq([lambda: self._ch('0'), lambda: self._not(self._digit_), lambda: self._succeed('0')])