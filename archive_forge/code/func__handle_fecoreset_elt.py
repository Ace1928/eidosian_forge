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
def _handle_fecoreset_elt(self, elt):
    """Load fe coreset info from xml."""
    info = self._load_xml_attributes(AttrDict(), elt)
    tmp = []
    for sub in elt:
        tmp.append(self._load_xml_attributes(AttrDict(), sub))
    return tmp