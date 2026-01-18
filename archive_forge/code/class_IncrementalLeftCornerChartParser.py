from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class IncrementalLeftCornerChartParser(IncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        if not grammar.is_nonempty():
            raise ValueError('IncrementalLeftCornerParser only works for grammars without empty productions.')
        IncrementalChartParser.__init__(self, grammar, LC_INCREMENTAL_STRATEGY, **parser_args)