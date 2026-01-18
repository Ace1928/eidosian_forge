from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class ScannerRule(CompleteFundamentalRule):
    _fundamental_rule = CompleteFundamentalRule()

    def apply(self, chart, grammar, edge):
        if isinstance(edge, LeafEdge):
            yield from self._fundamental_rule.apply(chart, grammar, edge)