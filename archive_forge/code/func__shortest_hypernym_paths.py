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
def _shortest_hypernym_paths(self, simulate_root):
    if self._name == '*ROOT*':
        return {self: 0}
    queue = deque([(self, 0)])
    path = {}
    while queue:
        s, depth = queue.popleft()
        if s in path:
            continue
        path[s] = depth
        depth += 1
        queue.extend(((hyp, depth) for hyp in s._hypernyms()))
        queue.extend(((hyp, depth) for hyp in s._instance_hypernyms()))
    if simulate_root:
        fake_synset = Synset(None)
        fake_synset._name = '*ROOT*'
        path[fake_synset] = max(path.values()) + 1
    return path