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
def _related(self, relation_symbol, sort=True):
    get_synset = self._wordnet_corpus_reader.synset_from_pos_and_offset
    if relation_symbol not in self._pointers:
        return []
    pointer_tuples = self._pointers[relation_symbol]
    r = [get_synset(pos, offset) for pos, offset in pointer_tuples]
    if sort:
        r.sort()
    return r