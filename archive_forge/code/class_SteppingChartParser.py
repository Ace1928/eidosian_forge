import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class SteppingChartParser(ChartParser):
    """
    A ``ChartParser`` that allows you to step through the parsing
    process, adding a single edge at a time.  It also allows you to
    change the parser's strategy or grammar midway through parsing a
    text.

    The ``initialize`` method is used to start parsing a text.  ``step``
    adds a single edge to the chart.  ``set_strategy`` changes the
    strategy used by the chart parser.  ``parses`` returns the set of
    parses that has been found by the chart parser.

    :ivar _restart: Records whether the parser's strategy, grammar,
        or chart has been changed.  If so, then ``step`` must restart
        the parsing algorithm.
    """

    def __init__(self, grammar, strategy=[], trace=0):
        self._chart = None
        self._current_chartrule = None
        self._restart = False
        ChartParser.__init__(self, grammar, strategy, trace)

    def initialize(self, tokens):
        """Begin parsing the given tokens."""
        self._chart = Chart(list(tokens))
        self._restart = True

    def step(self):
        """
        Return a generator that adds edges to the chart, one at a
        time.  Each time the generator is resumed, it adds a single
        edge and yields that edge.  If no more edges can be added,
        then it yields None.

        If the parser's strategy, grammar, or chart is changed, then
        the generator will continue adding edges using the new
        strategy, grammar, or chart.

        Note that this generator never terminates, since the grammar
        or strategy might be changed to values that would add new
        edges.  Instead, it yields None when no more edges can be
        added with the current strategy and grammar.
        """
        if self._chart is None:
            raise ValueError('Parser must be initialized first')
        while True:
            self._restart = False
            w = 50 // (self._chart.num_leaves() + 1)
            for e in self._parse():
                if self._trace > 1:
                    print(self._current_chartrule)
                if self._trace > 0:
                    print(self._chart.pretty_format_edge(e, w))
                yield e
                if self._restart:
                    break
            else:
                yield None

    def _parse(self):
        """
        A generator that implements the actual parsing algorithm.
        ``step`` iterates through this generator, and restarts it
        whenever the parser's strategy, grammar, or chart is modified.
        """
        chart = self._chart
        grammar = self._grammar
        edges_added = 1
        while edges_added > 0:
            edges_added = 0
            for rule in self._strategy:
                self._current_chartrule = rule
                for e in rule.apply_everywhere(chart, grammar):
                    edges_added += 1
                    yield e

    def strategy(self):
        """Return the strategy used by this parser."""
        return self._strategy

    def grammar(self):
        """Return the grammar used by this parser."""
        return self._grammar

    def chart(self):
        """Return the chart that is used by this parser."""
        return self._chart

    def current_chartrule(self):
        """Return the chart rule used to generate the most recent edge."""
        return self._current_chartrule

    def parses(self, tree_class=Tree):
        """Return the parse trees currently contained in the chart."""
        return self._chart.parses(self._grammar.start(), tree_class)

    def set_strategy(self, strategy):
        """
        Change the strategy that the parser uses to decide which edges
        to add to the chart.

        :type strategy: list(ChartRuleI)
        :param strategy: A list of rules that should be used to decide
            what edges to add to the chart.
        """
        if strategy == self._strategy:
            return
        self._strategy = strategy[:]
        self._restart = True

    def set_grammar(self, grammar):
        """Change the grammar used by the parser."""
        if grammar is self._grammar:
            return
        self._grammar = grammar
        self._restart = True

    def set_chart(self, chart):
        """Load a given chart into the chart parser."""
        if chart is self._chart:
            return
        self._chart = chart
        self._restart = True

    def parse(self, tokens, tree_class=Tree):
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        self.initialize(tokens)
        for e in self.step():
            if e is None:
                break
        return self.parses(tree_class=tree_class)