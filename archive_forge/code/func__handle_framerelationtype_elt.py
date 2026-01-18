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
def _handle_framerelationtype_elt(self, elt, *args):
    """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
    info = self._load_xml_attributes(AttrDict(), elt)
    info['_type'] = 'framerelationtype'
    info['frameRelations'] = PrettyList()
    for sub in elt:
        if sub.tag.endswith('frameRelation'):
            frel = self._handle_framerelation_elt(sub)
            frel['type'] = info
            for ferel in frel.feRelations:
                ferel['type'] = info
            info['frameRelations'].append(frel)
    return info