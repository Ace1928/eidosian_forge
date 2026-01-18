import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class LeftCornerChartParser(ChartParser):

    def __init__(self, grammar, **parser_args):
        if not grammar.is_nonempty():
            raise ValueError('LeftCornerParser only works for grammars without empty productions.')
        ChartParser.__init__(self, grammar, LC_STRATEGY, **parser_args)