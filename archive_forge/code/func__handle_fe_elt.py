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
def _handle_fe_elt(self, elt):
    feinfo = self._load_xml_attributes(AttrDict(), elt)
    feinfo['_type'] = 'fe'
    feinfo['definition'] = ''
    feinfo['definitionMarkup'] = ''
    feinfo['semType'] = None
    feinfo['requiresFE'] = None
    feinfo['excludesFE'] = None
    for sub in elt:
        if sub.tag.endswith('definition'):
            feinfo['definitionMarkup'] = sub.text
            feinfo['definition'] = self._strip_tags(sub.text)
        elif sub.tag.endswith('semType'):
            stinfo = self._load_xml_attributes(AttrDict(), sub)
            feinfo['semType'] = self.semtype(stinfo.ID)
        elif sub.tag.endswith('requiresFE'):
            feinfo['requiresFE'] = self._load_xml_attributes(AttrDict(), sub)
        elif sub.tag.endswith('excludesFE'):
            feinfo['excludesFE'] = self._load_xml_attributes(AttrDict(), sub)
    return feinfo