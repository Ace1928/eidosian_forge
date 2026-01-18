from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Dict
from nltk.internals import deprecated, overridden
from nltk.metrics import ConfusionMatrix, accuracy
from nltk.tag.util import untag
class FeaturesetTaggerI(TaggerI):
    """
    A tagger that requires tokens to be ``featuresets``.  A featureset
    is a dictionary that maps from feature names to feature
    values.  See ``nltk.classify`` for more information about features
    and featuresets.
    """