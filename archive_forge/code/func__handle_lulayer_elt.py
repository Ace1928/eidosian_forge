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
def _handle_lulayer_elt(self, elt):
    """Load a layer from an annotation set"""
    layer = self._load_xml_attributes(AttrDict(), elt)
    layer['_type'] = 'lulayer'
    layer['label'] = []
    for sub in elt:
        if sub.tag.endswith('label'):
            l = self._load_xml_attributes(AttrDict(), sub)
            if l is not None:
                layer['label'].append(l)
    return layer