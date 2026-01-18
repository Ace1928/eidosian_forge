import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _read_pcfg_production(input):
    """
    Return a list of PCFG ``ProbabilisticProductions``.
    """
    return _read_production(input, standard_nonterm_parser, probabilistic=True)