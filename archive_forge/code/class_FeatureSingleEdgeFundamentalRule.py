from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureSingleEdgeFundamentalRule(SingleEdgeFundamentalRule):
    """
    A specialized version of the completer / single edge fundamental rule
    that operates on nonterminals whose symbols are ``FeatStructNonterminal``.
    Rather than simply comparing the nonterminals for equality, they are
    unified.
    """
    _fundamental_rule = FeatureFundamentalRule()

    def _apply_complete(self, chart, grammar, right_edge):
        fr = self._fundamental_rule
        for left_edge in chart.select(end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()):
            yield from fr.apply(chart, grammar, left_edge, right_edge)

    def _apply_incomplete(self, chart, grammar, left_edge):
        fr = self._fundamental_rule
        for right_edge in chart.select(start=left_edge.end(), is_complete=True, lhs=left_edge.nextsym()):
            yield from fr.apply(chart, grammar, left_edge, right_edge)