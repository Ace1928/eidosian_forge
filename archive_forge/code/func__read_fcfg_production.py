import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _read_fcfg_production(input, fstruct_reader):
    """
    Return a list of feature-based ``Productions``.
    """
    return _read_production(input, fstruct_reader)