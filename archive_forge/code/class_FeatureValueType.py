import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
@total_ordering
class FeatureValueType:
    """
    A helper class for ``FeatureGrammars``, designed to be different
    from ordinary strings.  This is to stop the ``FeatStruct``
    ``FOO[]`` from being compare equal to the terminal "FOO".
    """

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return '<%s>' % self._value

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other._value

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, FeatureValueType):
            raise_unorderable_types('<', self, other)
        return self._value < other._value

    def __hash__(self):
        return hash(self._value)