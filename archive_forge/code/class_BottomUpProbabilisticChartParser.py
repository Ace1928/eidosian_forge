import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class BottomUpProbabilisticChartParser(ParserI):
    """
    An abstract bottom-up parser for ``PCFG`` grammars that uses a ``Chart`` to
    record partial results.  ``BottomUpProbabilisticChartParser`` maintains
    a queue of edges that can be added to the chart.  This queue is
    initialized with edges for each token in the text that is being
    parsed.  ``BottomUpProbabilisticChartParser`` inserts these edges into
    the chart one at a time, starting with the most likely edges, and
    proceeding to less likely edges.  For each edge that is added to
    the chart, it may become possible to insert additional edges into
    the chart; these are added to the queue.  This process continues
    until enough complete parses have been generated, or until the
    queue is empty.

    The sorting order for the queue is not specified by
    ``BottomUpProbabilisticChartParser``.  Different sorting orders will
    result in different search strategies.  The sorting order for the
    queue is defined by the method ``sort_queue``; subclasses are required
    to provide a definition for this method.

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """

    def __init__(self, grammar, beam_size=0, trace=0):
        """
        Create a new ``BottomUpProbabilisticChartParser``, that uses
        ``grammar`` to parse texts.

        :type grammar: PCFG
        :param grammar: The grammar used to parse texts.
        :type beam_size: int
        :param beam_size: The maximum length for the parser's edge queue.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        """
        if not isinstance(grammar, PCFG):
            raise ValueError('The grammar must be probabilistic PCFG')
        self._grammar = grammar
        self.beam_size = beam_size
        self._trace = trace

    def grammar(self):
        return self._grammar

    def trace(self, trace=2):
        """
        Set the level of tracing output that should be generated when
        parsing a text.

        :type trace: int
        :param trace: The trace level.  A trace level of ``0`` will
            generate no tracing output; and higher trace levels will
            produce more verbose tracing output.
        :rtype: None
        """
        self._trace = trace

    def parse(self, tokens):
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        grammar = self._grammar
        bu_init = ProbabilisticBottomUpInitRule()
        bu = ProbabilisticBottomUpPredictRule()
        fr = SingleEdgeProbabilisticFundamentalRule()
        queue = []
        for edge in bu_init.apply(chart, grammar):
            if self._trace > 1:
                print('  %-50s [%s]' % (chart.pretty_format_edge(edge, width=2), edge.prob()))
            queue.append(edge)
        while len(queue) > 0:
            self.sort_queue(queue, chart)
            if self.beam_size:
                self._prune(queue, chart)
            edge = queue.pop()
            if self._trace > 0:
                print('  %-50s [%s]' % (chart.pretty_format_edge(edge, width=2), edge.prob()))
            queue.extend(bu.apply(chart, grammar, edge))
            queue.extend(fr.apply(chart, grammar, edge))
        parses = list(chart.parses(grammar.start(), ProbabilisticTree))
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        for parse in parses:
            self._setprob(parse, prod_probs)
        parses.sort(reverse=True, key=lambda tree: tree.prob())
        return iter(parses)

    def _setprob(self, tree, prod_probs):
        if tree.prob() is not None:
            return
        lhs = Nonterminal(tree.label())
        rhs = []
        for child in tree:
            if isinstance(child, Tree):
                rhs.append(Nonterminal(child.label()))
            else:
                rhs.append(child)
        prob = prod_probs[lhs, tuple(rhs)]
        for child in tree:
            if isinstance(child, Tree):
                self._setprob(child, prod_probs)
                prob *= child.prob()
        tree.set_prob(prob)

    def sort_queue(self, queue, chart):
        """
        Sort the given queue of ``Edge`` objects, placing the edge that should
        be tried first at the beginning of the queue.  This method
        will be called after each ``Edge`` is added to the queue.

        :param queue: The queue of ``Edge`` objects to sort.  Each edge in
            this queue is an edge that could be added to the chart by
            the fundamental rule; but that has not yet been added.
        :type queue: list(Edge)
        :param chart: The chart being used to parse the text.  This
            chart can be used to provide extra information for sorting
            the queue.
        :type chart: Chart
        :rtype: None
        """
        raise NotImplementedError()

    def _prune(self, queue, chart):
        """Discard items in the queue if the queue is longer than the beam."""
        if len(queue) > self.beam_size:
            split = len(queue) - self.beam_size
            if self._trace > 2:
                for edge in queue[:split]:
                    print('  %-50s [DISCARDED]' % chart.pretty_format_edge(edge, 2))
            del queue[:split]