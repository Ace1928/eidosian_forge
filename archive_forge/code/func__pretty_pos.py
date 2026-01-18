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
def _pretty_pos(aset):
    """
    Helper function for pretty-printing a sentence with its POS tags.

    :param aset: The POS annotation set of the sentence to be printed.
    :type sent: list(AttrDict)
    :return: The text of the sentence and its POS tags.
    :rtype: str
    """
    outstr = ''
    outstr += 'POS annotation set ({0.ID}) {0.POS_tagset} in sentence {0.sent.ID}:\n\n'.format(aset)
    overt = sorted(aset.POS)
    sent = aset.sent
    s0 = sent.text
    s1 = ''
    s2 = ''
    i = 0
    adjust = 0
    for j, k, lbl in overt:
        assert j >= i, ('Overlapping targets?', (j, k, lbl))
        s1 += ' ' * (j - i) + '-' * (k - j)
        if len(lbl) > k - j:
            amt = len(lbl) - (k - j)
            s0 = s0[:k + adjust] + '~' * amt + s0[k + adjust:]
            s1 = s1[:k + adjust] + ' ' * amt + s1[k + adjust:]
            adjust += amt
        s2 += ' ' * (j - i) + lbl.ljust(k - j)
        i = k
    long_lines = [s0, s1, s2]
    outstr += '\n\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' '))).replace('~', ' ')
    outstr += '\n'
    return outstr