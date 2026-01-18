import sys
from breezy import tests
from breezy.tests import features
class _BadSecondHash(_Hashable):

    def __init__(self, the_hash):
        _Hashable.__init__(self, the_hash)
        self._first = True

    def __hash__(self):
        if self._first:
            self._first = False
            return self.hash
        else:
            raise ValueError('I can only be hashed once.')