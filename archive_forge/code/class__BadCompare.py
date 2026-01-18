import sys
from breezy import tests
from breezy.tests import features
class _BadCompare(_Hashable):

    def __eq__(self, other):
        raise RuntimeError('I refuse to play nice')
    __hash__ = _Hashable.__hash__