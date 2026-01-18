import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class AbstractChartRule(ChartRuleI):
    """
    An abstract base class for chart rules.  ``AbstractChartRule``
    provides:

    - A default implementation for ``apply``.
    - A default implementation for ``apply_everywhere``,
      (Currently, this implementation assumes that ``NUM_EDGES <= 3``.)
    - A default implementation for ``__str__``, which returns a
      name based on the rule's class name.
    """

    def apply(self, chart, grammar, *edges):
        raise NotImplementedError()

    def apply_everywhere(self, chart, grammar):
        if self.NUM_EDGES == 0:
            yield from self.apply(chart, grammar)
        elif self.NUM_EDGES == 1:
            for e1 in chart:
                yield from self.apply(chart, grammar, e1)
        elif self.NUM_EDGES == 2:
            for e1 in chart:
                for e2 in chart:
                    yield from self.apply(chart, grammar, e1, e2)
        elif self.NUM_EDGES == 3:
            for e1 in chart:
                for e2 in chart:
                    for e3 in chart:
                        yield from self.apply(chart, grammar, e1, e2, e3)
        else:
            raise AssertionError('NUM_EDGES>3 is not currently supported')

    def __str__(self):
        return re.sub('([a-z])([A-Z])', '\\1 \\2', self.__class__.__name__)