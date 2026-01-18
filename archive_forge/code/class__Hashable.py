import sys
from breezy import tests
from breezy.tests import features
class _Hashable:
    """A simple object which has a fixed hash value.

    We could have used an 'int', but it turns out that Int objects don't
    implement tp_richcompare in Python 2.
    """

    def __init__(self, the_hash):
        self.hash = the_hash

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, _Hashable):
            return NotImplemented
        return other.hash == self.hash