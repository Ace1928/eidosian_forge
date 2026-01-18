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
def _pretty_exemplars(exemplars, lu):
    """
    Helper function for pretty-printing a list of exemplar sentences for a lexical unit.

    :param sent: The list of exemplar sentences to be printed.
    :type sent: list(AttrDict)
    :return: An index of the text of the exemplar sentences.
    :rtype: str
    """
    outstr = ''
    outstr += 'exemplar sentences for {0.name} in {0.frame.name}:\n\n'.format(lu)
    for i, sent in enumerate(exemplars):
        outstr += f'[{i}] {sent.text}\n'
    outstr += '\n'
    return outstr