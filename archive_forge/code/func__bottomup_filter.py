import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def _bottomup_filter(grammar, nexttoken, rhs, dot=0):
    if len(rhs) <= dot + 1:
        return True
    _next = rhs[dot + 1]
    if is_terminal(_next):
        return nexttoken == _next
    else:
        return grammar.is_leftcorner(_next, nexttoken)