from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureTopDownChartParser(FeatureChartParser):

    def __init__(self, grammar, **parser_args):
        FeatureChartParser.__init__(self, grammar, TD_FEATURE_STRATEGY, **parser_args)