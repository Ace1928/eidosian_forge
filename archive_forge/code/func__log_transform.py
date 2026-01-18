import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _log_transform(self, probability):
    """Return log transform of the given probability dictionary (PRIVATE).

        When calculating the Viterbi equation, add logs of probabilities rather
        than multiplying probabilities, to avoid underflow errors. This method
        returns a new dictionary with the same keys as the given dictionary
        and log-transformed values.
        """
    log_prob = copy.copy(probability)
    for key in log_prob:
        prob = log_prob[key]
        if prob > 0:
            log_prob[key] = math.log(log_prob[key])
        else:
            log_prob[key] = -math.inf
    return log_prob