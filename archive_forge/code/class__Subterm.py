from __future__ import print_function
from patsy.util import no_pickling
class _Subterm(object):
    """Also immutable."""

    def __init__(self, efactors):
        self.efactors = frozenset(efactors)

    def can_absorb(self, other):
        return len(self.efactors) - len(other.efactors) == 1 and self.efactors.issuperset(other.efactors)

    def absorb(self, other):
        diff = self.efactors.difference(other.efactors)
        assert len(diff) == 1
        efactor = list(diff)[0]
        assert not efactor.includes_intercept
        new_factors = set(other.efactors)
        new_factors.add(_ExpandedFactor(True, efactor.factor))
        return _Subterm(new_factors)

    def __hash__(self):
        return hash((_Subterm, self.efactors))

    def __eq__(self, other):
        return isinstance(other, _Subterm) and self.efactors == self.efactors

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(self.efactors))
    __getstate__ = no_pickling