import unicodedata
def _id_continue__c4__s1_(self):
    v = self._is_unicat(self._get('x'), 'Mc')
    if v:
        self._succeed(v)
    else:
        self._fail()