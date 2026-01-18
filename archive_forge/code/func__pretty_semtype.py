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
def _pretty_semtype(st):
    """
    Helper function for pretty-printing a semantic type.

    :param st: The semantic type to be printed.
    :type st: AttrDict
    :return: A nicely formatted string representation of the semantic type.
    :rtype: str
    """
    semkeys = st.keys()
    if len(semkeys) == 1:
        return '<None>'
    outstr = ''
    outstr += 'semantic type ({0.ID}): {0.name}\n'.format(st)
    if 'abbrev' in semkeys:
        outstr += f'[abbrev] {st.abbrev}\n'
    if 'definition' in semkeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(st.definition, '  ')
    outstr += f'[rootType] {st.rootType.name}({st.rootType.ID})\n'
    if st.superType is None:
        outstr += '[superType] <None>\n'
    else:
        outstr += f'[superType] {st.superType.name}({st.superType.ID})\n'
    outstr += f'[subTypes] {len(st.subTypes)} subtypes\n'
    outstr += '  ' + ', '.join((f'{x.name}({x.ID})' for x in st.subTypes)) + '\n' * (len(st.subTypes) > 0)
    return outstr