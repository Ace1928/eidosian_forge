from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _create_hmm_tagger(states, symbols, A, B, pi):

    def pd(values, samples):
        d = dict(zip(samples, values))
        return DictionaryProbDist(d)

    def cpd(array, conditions, samples):
        d = {}
        for values, condition in zip(array, conditions):
            d[condition] = pd(values, samples)
        return DictionaryConditionalProbDist(d)
    A = cpd(A, states, states)
    B = cpd(B, states, symbols)
    pi = pd(pi, states)
    return HiddenMarkovModelTagger(symbols=symbols, states=states, transitions=A, outputs=B, priors=pi)