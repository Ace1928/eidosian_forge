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
def _handle_fulltextannotationset_elt(self, elt, is_pos=False):
    """Load information from the given 'annotationSet' element. Each
        'annotationSet' contains several "layer" elements."""
    info = self._handle_luannotationset_elt(elt, is_pos=is_pos)
    if not is_pos:
        info['_type'] = 'fulltext_annotationset'
        if 'cxnID' not in info:
            info['LU'] = self.lu(info.luID, luName=info.luName, frameID=info.frameID, frameName=info.frameName)
            info['frame'] = info.LU.frame
    return info