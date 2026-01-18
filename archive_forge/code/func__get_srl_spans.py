import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _get_srl_spans(self, grid):
    """
        list of list of (start, end), tag) tuples
        """
    if self._srl_includes_roleset:
        predicates = self._get_column(grid, self._colmap['srl'] + 1)
        start_col = self._colmap['srl'] + 2
    else:
        predicates = self._get_column(grid, self._colmap['srl'])
        start_col = self._colmap['srl'] + 1
    num_preds = len([p for p in predicates if p != '-'])
    spanlists = []
    for i in range(num_preds):
        col = self._get_column(grid, start_col + i)
        spanlist = []
        stack = []
        for wordnum, srl_tag in enumerate(col):
            left, right = srl_tag.split('*')
            for tag in left.split('('):
                if tag:
                    stack.append((tag, wordnum))
            for i in range(right.count(')')):
                tag, start = stack.pop()
                spanlist.append(((start, wordnum + 1), tag))
        spanlists.append(spanlist)
    return spanlists