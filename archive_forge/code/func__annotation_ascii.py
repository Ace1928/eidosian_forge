import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _annotation_ascii(sent):
    """
    Given a sentence or FE annotation set, construct the width-limited string showing
    an ASCII visualization of the sentence's annotations, calling either
    _annotation_ascii_frames() or _annotation_ascii_FEs() as appropriate.
    This will be attached as a method to appropriate AttrDict instances
    and called in the full pretty-printing of the instance.
    """
    if sent._type == 'fulltext_sentence' or ('annotationSet' in sent and len(sent.annotationSet) > 2):
        return _annotation_ascii_frames(sent)
    else:
        return _annotation_ascii_FEs(sent)