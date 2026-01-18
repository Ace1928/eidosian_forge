import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatureValueUnion(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that represents the union of two or more
    ``FeatureValueSet`` or ``Variable``.
    """

    def __new__(cls, values):
        values = _flatten(values, FeatureValueUnion)
        if sum((isinstance(v, Variable) for v in values)) == 0:
            values = _flatten(values, FeatureValueSet)
            return FeatureValueSet(values)
        if len(values) == 1:
            return list(values)[0]
        return frozenset.__new__(cls, values)

    def __repr__(self):
        return '{%s}' % '+'.join(sorted((f'{b}' for b in self)))