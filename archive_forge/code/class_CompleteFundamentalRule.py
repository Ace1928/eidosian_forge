from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class CompleteFundamentalRule(SingleEdgeFundamentalRule):

    def _apply_incomplete(self, chart, grammar, left_edge):
        end = left_edge.end()
        for right_edge in chart.select(start=end, end=end, is_complete=True, lhs=left_edge.nextsym()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge