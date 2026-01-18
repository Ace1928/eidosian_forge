from __future__ import print_function
from patsy.util import no_pickling
class _ExpandedFactor(object):
    """A factor, with an additional annotation for whether it is coded
    full-rank (includes_intercept=True) or not.

    These objects are treated as immutable."""

    def __init__(self, includes_intercept, factor):
        self.includes_intercept = includes_intercept
        self.factor = factor

    def __hash__(self):
        return hash((_ExpandedFactor, self.includes_intercept, self.factor))

    def __eq__(self, other):
        return isinstance(other, _ExpandedFactor) and other.includes_intercept == self.includes_intercept and (other.factor == self.factor)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.includes_intercept:
            suffix = '+'
        else:
            suffix = '-'
        return '%r%s' % (self.factor, suffix)
    __getstate__ = no_pickling