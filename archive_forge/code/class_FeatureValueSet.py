import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatureValueSet(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that is a set of other base feature values.
    FeatureValueSet implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueSet`` is immutable.
    """

    def __repr__(self):
        if len(self) == 0:
            return '{/}'
        return '{%s}' % ', '.join(sorted((f'{b}' for b in self)))
    __str__ = __repr__