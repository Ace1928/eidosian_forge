import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@staticmethod
def _dunning_log_likelihood(count_a, count_b, count_ab, N):
    """
        A function that calculates the modified Dunning log-likelihood
        ratio scores for abbreviation candidates.  The details of how
        this works is available in the paper.
        """
    p1 = count_b / N
    p2 = 0.99
    null_hypo = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
    alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)
    likelihood = null_hypo - alt_hypo
    return -2.0 * likelihood