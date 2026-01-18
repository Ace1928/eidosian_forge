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
def _handle_framerelation_elt(self, elt):
    """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
    info = self._load_xml_attributes(AttrDict(), elt)
    assert info['superFrameName'] != info['subFrameName'], (elt, info)
    info['_type'] = 'framerelation'
    info['feRelations'] = PrettyList()
    for sub in elt:
        if sub.tag.endswith('FERelation'):
            ferel = self._handle_elt(sub)
            ferel['_type'] = 'ferelation'
            ferel['frameRelation'] = info
            info['feRelations'].append(ferel)
    return info