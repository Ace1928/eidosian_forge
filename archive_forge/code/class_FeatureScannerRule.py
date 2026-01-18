from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FeatureScannerRule(ScannerRule):
    _fundamental_rule = FeatureCompleteFundamentalRule()