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
def _handle_lusentence_elt(self, elt):
    """Load a sentence from a subcorpus of an LU from xml."""
    info = self._load_xml_attributes(AttrDict(), elt)
    info['_type'] = 'lusentence'
    info['annotationSet'] = []
    info['_ascii'] = types.MethodType(_annotation_ascii, info)
    for sub in elt:
        if sub.tag.endswith('text'):
            info['text'] = self._strip_tags(sub.text)
        elif sub.tag.endswith('annotationSet'):
            annset = self._handle_luannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
            if annset is not None:
                assert annset.status == 'UNANN' or 'FE' in annset, annset
                if annset.status != 'UNANN':
                    info['frameAnnotation'] = annset
                for k in ('Target', 'FE', 'FE2', 'FE3', 'GF', 'PT', 'POS', 'POS_tagset', 'Other', 'Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
                    if k in annset:
                        info[k] = annset[k]
                info['annotationSet'].append(annset)
                annset['sent'] = info
                annset['text'] = info.text
    return info