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
def _pretty_fulltext_sentence(sent):
    """
    Helper function for pretty-printing an annotated sentence from a full-text document.

    :param sent: The sentence to be printed.
    :type sent: list(AttrDict)
    :return: The text of the sentence with annotation set indices on frame targets.
    :rtype: str
    """
    outstr = ''
    outstr += 'full-text sentence ({0.ID}) in {1}:\n\n'.format(sent, sent.doc.get('name', sent.doc.description))
    outstr += f'\n[POS] {len(sent.POS)} tags\n'
    outstr += f'\n[POS_tagset] {sent.POS_tagset}\n\n'
    outstr += '[text] + [annotationSet]\n\n'
    outstr += sent._ascii()
    outstr += '\n'
    return outstr