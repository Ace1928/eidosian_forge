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
def _load_lemma_pos_offset_map(self):
    for suffix in self._FILEMAP.values():
        with self.open('index.%s' % suffix) as fp:
            for i, line in enumerate(fp):
                if line.startswith(' '):
                    continue
                _iter = iter(line.split())

                def _next_token():
                    return next(_iter)
                try:
                    lemma = _next_token()
                    pos = _next_token()
                    n_synsets = int(_next_token())
                    assert n_synsets > 0
                    n_pointers = int(_next_token())
                    [_next_token() for _ in range(n_pointers)]
                    n_senses = int(_next_token())
                    assert n_synsets == n_senses
                    _next_token()
                    synset_offsets = [int(_next_token()) for _ in range(n_synsets)]
                except (AssertionError, ValueError) as e:
                    tup = ('index.%s' % suffix, i + 1, e)
                    raise WordNetError('file %s, line %i: %s' % tup) from e
                self._lemma_pos_offset_map[lemma][pos] = synset_offsets
                if pos == ADJ:
                    self._lemma_pos_offset_map[lemma][ADJ_SAT] = synset_offsets