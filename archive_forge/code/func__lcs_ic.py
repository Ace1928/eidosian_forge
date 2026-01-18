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
def _lcs_ic(synset1, synset2, ic, verbose=False):
    """
    Get the information content of the least common subsumer that has
    the highest information content value.  If two nodes have no
    explicit common subsumer, assume that they share an artificial
    root node that is the hypernym of all explicit roots.

    :type synset1: Synset
    :param synset1: First input synset.
    :type synset2: Synset
    :param synset2: Second input synset.  Must be the same part of
    speech as the first synset.
    :type  ic: dict
    :param ic: an information content object (as returned by ``load_ic()``).
    :return: The information content of the two synsets and their most
    informative subsumer
    """
    if synset1._pos != synset2._pos:
        raise WordNetError('Computing the least common subsumer requires %s and %s to have the same part of speech.' % (synset1, synset2))
    ic1 = information_content(synset1, ic)
    ic2 = information_content(synset2, ic)
    subsumers = synset1.common_hypernyms(synset2)
    if len(subsumers) == 0:
        subsumer_ic = 0
    else:
        subsumer_ic = max((information_content(s, ic) for s in subsumers))
    if verbose:
        print('> LCS Subsumer by content:', subsumer_ic)
    return (ic1, ic2, subsumer_ic)