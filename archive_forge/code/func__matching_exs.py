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
def _matching_exs():
    for f in frames:
        fes = fes2 = None
        if fe is not None:
            fes = {ffe for ffe in f.FE.keys() if re.search(fe, ffe, re.I)} if isinstance(fe, str) else {fe.name}
            if fe2 is not None:
                fes2 = {ffe for ffe in f.FE.keys() if re.search(fe2, ffe, re.I)} if isinstance(fe2, str) else {fe2.name}
        for lu in lusByFrame[f.name] if luNamePattern is not None else f.lexUnit.values():
            for ex in lu.exemplars:
                if (fes is None or self._exemplar_of_fes(ex, fes)) and (fes2 is None or self._exemplar_of_fes(ex, fes2)):
                    yield ex