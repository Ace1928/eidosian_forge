import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class BottomUpLeftCornerChartParser(ChartParser):
    """
    A ``ChartParser`` using a bottom-up left-corner parsing strategy.
    This strategy is often more efficient than standard bottom-up.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        ChartParser.__init__(self, grammar, BU_LC_STRATEGY, **parser_args)