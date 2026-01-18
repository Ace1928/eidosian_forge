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
@deprecated('Use public method synset_from_pos_and_offset() instead')
def _synset_from_pos_and_offset(self, *args, **kwargs):
    """
        Hack to help people like the readers of
        https://stackoverflow.com/a/27145655/1709587
        who were using this function before it was officially a public method
        """
    return self.synset_from_pos_and_offset(*args, **kwargs)