import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
@staticmethod
def _default_context(tokens, i):
    """One left token and one right token, normalized to lowercase"""
    left = tokens[i - 1].lower() if i != 0 else '*START*'
    right = tokens[i + 1].lower() if i != len(tokens) - 1 else '*END*'
    return (left, right)