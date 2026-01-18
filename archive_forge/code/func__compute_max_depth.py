import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def _compute_max_depth(self, pos, simulate_root):
    """
        Compute the max depth for the given part of speech.  This is
        used by the lch similarity metric.
        """
    depth = 0
    for ii in self.all_synsets(pos):
        try:
            depth = max(depth, ii.max_depth())
        except RuntimeError:
            print(ii)
    if simulate_root:
        depth += 1
    self._max_depth[pos] = depth