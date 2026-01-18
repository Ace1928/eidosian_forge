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
def _handle_fulltextannotation_elt(self, elt):
    """Load full annotation info for a document from its xml
        file. The main element (fullTextAnnotation) contains a 'header'
        element (which we ignore here) and a bunch of 'sentence'
        elements."""
    info = AttrDict()
    info['_type'] = 'fulltext_annotation'
    info['sentence'] = []
    for sub in elt:
        if sub.tag.endswith('header'):
            continue
        elif sub.tag.endswith('sentence'):
            s = self._handle_fulltext_sentence_elt(sub)
            s.doc = info
            info['sentence'].append(s)
    return info