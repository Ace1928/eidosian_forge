import unicodedata
def _num_literal_(self):
    self._choose([self._num_literal__c0_, self._num_literal__c1_, self._num_literal__c2_, self._hex_literal_, self._num_literal__c4_, self._num_literal__c5_])