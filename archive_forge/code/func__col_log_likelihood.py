import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@staticmethod
def _col_log_likelihood(count_a, count_b, count_ab, N):
    """
        A function that will just compute log-likelihood estimate, in
        the original paper it's described in algorithm 6 and 7.

        This *should* be the original Dunning log-likelihood values,
        unlike the previous log_l function where it used modified
        Dunning log-likelihood values
        """
    p = count_b / N
    p1 = count_ab / count_a
    try:
        p2 = (count_b - count_ab) / (N - count_a)
    except ZeroDivisionError:
        p2 = 1
    try:
        summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)
    except ValueError:
        summand1 = 0
    try:
        summand2 = (count_b - count_ab) * math.log(p) + (N - count_a - count_b + count_ab) * math.log(1.0 - p)
    except ValueError:
        summand2 = 0
    if count_a == count_ab or p1 <= 0 or p1 >= 1:
        summand3 = 0
    else:
        summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
    if count_b == count_ab or p2 <= 0 or p2 >= 1:
        summand4 = 0
    else:
        summand4 = (count_b - count_ab) * math.log(p2) + (N - count_a - count_b + count_ab) * math.log(1.0 - p2)
    likelihood = summand1 + summand2 - summand3 - summand4
    return -2.0 * likelihood