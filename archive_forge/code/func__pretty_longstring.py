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
def _pretty_longstring(defstr, prefix='', wrap_at=65):
    """
    Helper function for pretty-printing a long string.

    :param defstr: The string to be printed.
    :type defstr: str
    :return: A nicely formatted string representation of the long string.
    :rtype: str
    """
    outstr = ''
    for line in textwrap.fill(defstr, wrap_at).split('\n'):
        outstr += prefix + line + '\n'
    return outstr