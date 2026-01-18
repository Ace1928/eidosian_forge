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
class WordNetICCorpusReader(CorpusReader):
    """
    A corpus reader for the WordNet information content corpus.
    """

    def __init__(self, root, fileids):
        CorpusReader.__init__(self, root, fileids, encoding='utf8')

    def ic(self, icfile):
        """
        Load an information content file from the wordnet_ic corpus
        and return a dictionary.  This dictionary has just two keys,
        NOUN and VERB, whose values are dictionaries that map from
        synsets to information content values.

        :type icfile: str
        :param icfile: The name of the wordnet_ic file (e.g. "ic-brown.dat")
        :return: An information content dictionary
        """
        ic = {}
        ic[NOUN] = defaultdict(float)
        ic[VERB] = defaultdict(float)
        with self.open(icfile) as fp:
            for num, line in enumerate(fp):
                if num == 0:
                    continue
                fields = line.split()
                offset = int(fields[0][:-1])
                value = float(fields[1])
                pos = _get_pos(fields[0])
                if len(fields) == 3 and fields[2] == 'ROOT':
                    ic[pos][0] += value
                if value != 0:
                    ic[pos][offset] = value
        return ic