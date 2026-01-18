import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _calculate_grammar_forms(self):
    """
        Pre-calculate of which form(s) the grammar is.
        """
    prods = self._productions
    self._is_lexical = all((p.is_lexical() for p in prods))
    self._is_nonlexical = all((p.is_nonlexical() for p in prods if len(p) != 1))
    self._min_len = min((len(p) for p in prods))
    self._max_len = max((len(p) for p in prods))
    self._all_unary_are_lexical = all((p.is_lexical() for p in prods if len(p) == 1))