from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class IncrementalBottomUpChartParser(IncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(self, grammar, BU_INCREMENTAL_STRATEGY, **parser_args)