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
def _buildframeindex(self):
    if not self._frel_idx:
        self._buildrelationindex()
    self._frame_idx = {}
    with XMLCorpusView(self.abspath('frameIndex.xml'), 'frameIndex/frame', self._handle_elt) as view:
        for f in view:
            self._frame_idx[f['ID']] = f