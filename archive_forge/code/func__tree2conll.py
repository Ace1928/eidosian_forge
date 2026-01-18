import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _tree2conll(self, tree, wordnum, words, pos, synt):
    assert isinstance(tree, Tree)
    if len(tree) == 1 and isinstance(tree[0], str):
        pos[wordnum] = tree.label()
        assert words[wordnum] == tree[0]
        return wordnum + 1
    elif len(tree) == 1 and isinstance(tree[0], tuple):
        assert len(tree[0]) == 2
        pos[wordnum], pos[wordnum] = tree[0]
        return wordnum + 1
    else:
        synt[wordnum] = f'({tree.label()}{synt[wordnum]}'
        for child in tree:
            wordnum = self._tree2conll(child, wordnum, words, pos, synt)
        synt[wordnum - 1] += ')'
        return wordnum