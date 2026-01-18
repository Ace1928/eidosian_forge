import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def _hd(self, i):
    try:
        return self.nodes[i]['head']
    except IndexError:
        return None