from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class EarleyChartParser(IncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(self, grammar, EARLEY_STRATEGY, **parser_args)