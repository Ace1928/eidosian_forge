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
def _handle_lusubcorpus_elt(self, elt):
    """Load a subcorpus of a lexical unit from the given xml."""
    sc = AttrDict()
    try:
        sc['name'] = elt.get('name')
    except AttributeError:
        return None
    sc['_type'] = 'lusubcorpus'
    sc['sentence'] = []
    for sub in elt:
        if sub.tag.endswith('sentence'):
            s = self._handle_lusentence_elt(sub)
            if s is not None:
                sc['sentence'].append(s)
    return sc