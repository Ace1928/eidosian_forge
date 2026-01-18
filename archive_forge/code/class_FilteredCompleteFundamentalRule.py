from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FilteredCompleteFundamentalRule(FilteredSingleEdgeFundamentalRule):

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            yield from self._apply_complete(chart, grammar, edge)