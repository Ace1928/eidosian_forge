from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _exhaustive_entropy(self, unlabeled_sequence):
    unlabeled_sequence = self._transform(unlabeled_sequence)
    T = len(unlabeled_sequence)
    N = len(self._states)
    labellings = [[state] for state in self._states]
    for t in range(T - 1):
        current = labellings
        labellings = []
        for labelling in current:
            for state in self._states:
                labellings.append(labelling + [state])
    log_probs = []
    for labelling in labellings:
        labeled_sequence = unlabeled_sequence[:]
        for t, label in enumerate(labelling):
            labeled_sequence[t] = (labeled_sequence[t][_TEXT], label)
        lp = self.log_probability(labeled_sequence)
        log_probs.append(lp)
    normalisation = _log_add(*log_probs)
    entropy = 0
    for lp in log_probs:
        lp -= normalisation
        entropy -= 2 ** lp * lp
    return entropy