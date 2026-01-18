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
def _annotation_ascii_FEs(sent):
    """
    ASCII string rendering of the sentence along with a single target and its FEs.
    Secondary and tertiary FE layers are included if present.
    'sent' can be an FE annotation set or an LU sentence with a single target.
    Line-wrapped to limit the display width.
    """
    feAbbrevs = OrderedDict()
    posspec = []
    posspec_separate = False
    for lyr in ('Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
        if lyr in sent and sent[lyr]:
            for a, b, lbl in sent[lyr]:
                if lbl == 'X':
                    continue
                if any((1 for x, y, felbl in sent.FE[0] if x <= a < y or a <= x < b)):
                    posspec_separate = True
                posspec.append((a, b, lbl.lower().replace('-', '')))
    if posspec_separate:
        POSSPEC = _annotation_ascii_FE_layer(posspec, {}, feAbbrevs)
    FE1 = _annotation_ascii_FE_layer(sorted(sent.FE[0] + (posspec if not posspec_separate else [])), sent.FE[1], feAbbrevs)
    FE2 = FE3 = None
    if 'FE2' in sent:
        FE2 = _annotation_ascii_FE_layer(sent.FE2[0], sent.FE2[1], feAbbrevs)
        if 'FE3' in sent:
            FE3 = _annotation_ascii_FE_layer(sent.FE3[0], sent.FE3[1], feAbbrevs)
    for i, j in sent.Target:
        FE1span, FE1name, FE1exp = FE1
        if len(FE1span) < j:
            FE1span += ' ' * (j - len(FE1span))
        if len(FE1name) < j:
            FE1name += ' ' * (j - len(FE1name))
            FE1[1] = FE1name
        FE1[0] = FE1span[:i] + FE1span[i:j].replace(' ', '*').replace('-', '=') + FE1span[j:]
    long_lines = [sent.text]
    if posspec_separate:
        long_lines.extend(POSSPEC[:2])
    long_lines.extend([FE1[0], FE1[1] + FE1[2]])
    if FE2:
        long_lines.extend([FE2[0], FE2[1] + FE2[2]])
        if FE3:
            long_lines.extend([FE3[0], FE3[1] + FE3[2]])
    long_lines.append('')
    outstr = '\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' ')))
    if feAbbrevs:
        outstr += '(' + ', '.join(('='.join(pair) for pair in feAbbrevs.items())) + ')'
        assert len(feAbbrevs) == len(dict(feAbbrevs)), 'Abbreviation clash'
    outstr += '\n'
    return outstr