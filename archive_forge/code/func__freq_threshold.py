import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _freq_threshold(self, fdist, threshold):
    """
        Returns a FreqDist containing only data with counts below a given
        threshold, as well as a mapping (None -> count_removed).
        """
    res = FreqDist()
    num_removed = 0
    for tok in fdist:
        count = fdist[tok]
        if count < threshold:
            num_removed += 1
        else:
            res[tok] += count
    res[None] += num_removed
    return res