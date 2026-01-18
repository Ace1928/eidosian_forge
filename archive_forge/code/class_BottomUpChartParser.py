import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class BottomUpChartParser(ChartParser):
    """
    A ``ChartParser`` using a bottom-up parsing strategy.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        if isinstance(grammar, PCFG):
            warnings.warn('BottomUpChartParser only works for CFG, use BottomUpProbabilisticChartParser instead', category=DeprecationWarning)
        ChartParser.__init__(self, grammar, BU_STRATEGY, **parser_args)