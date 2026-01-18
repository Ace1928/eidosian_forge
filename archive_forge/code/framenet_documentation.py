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

            # Look for boundary issues in markup. (Sometimes FEs are pluralized in definitions.)
            m = re.search(r'\w[<][^/]|[<][/][^>]+[>](s\w|[a-rt-z0-9])', data)
            if m:
                print('Markup boundary:', data[max(0,m.start(0)-10):m.end(0)+10].replace('\n',' '), file=sys.stderr)
            