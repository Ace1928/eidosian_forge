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
def _train_default_ngram_lm(self, tokenized_sents, n=3):
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_sents)
    model = MLE(order=n)
    model.fit(train_data, padded_sents)
    return model