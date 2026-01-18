import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class CachedTopDownPredictRule(TopDownPredictRule):
    """
    A cached version of ``TopDownPredictRule``.  After the first time
    this rule is applied to an edge with a given ``end`` and ``next``,
    it will not generate any more edges for edges with that ``end`` and
    ``next``.

    If ``chart`` or ``grammar`` are changed, then the cache is flushed.
    """

    def __init__(self):
        TopDownPredictRule.__init__(self)
        self._done = {}

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            return
        nextsym, index = (edge.nextsym(), edge.end())
        if not is_nonterminal(nextsym):
            return
        done = self._done.get((nextsym, index), (None, None))
        if done[0] is chart and done[1] is grammar:
            return
        for prod in grammar.productions(lhs=nextsym):
            if prod.rhs():
                first = prod.rhs()[0]
                if is_terminal(first):
                    if index >= chart.num_leaves() or first != chart.leaf(index):
                        continue
            new_edge = TreeEdge.from_production(prod, index)
            if chart.insert(new_edge, ()):
                yield new_edge
        self._done[nextsym, index] = (chart, grammar)