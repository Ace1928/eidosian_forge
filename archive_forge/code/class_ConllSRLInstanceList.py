import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
class ConllSRLInstanceList(list):
    """
    Set of instances for a single sentence
    """

    def __init__(self, tree, instances=()):
        self.tree = tree
        list.__init__(self, instances)

    def __str__(self):
        return self.pprint()

    def pprint(self, include_tree=False):
        for inst in self:
            if inst.tree != self.tree:
                raise ValueError('Tree mismatch!')
        if include_tree:
            words = self.tree.leaves()
            pos = [None] * len(words)
            synt = ['*'] * len(words)
            self._tree2conll(self.tree, 0, words, pos, synt)
        s = ''
        for i in range(len(words)):
            if include_tree:
                s += '%-20s ' % words[i]
                s += '%-8s ' % pos[i]
                s += '%15s*%-8s ' % tuple(synt[i].split('*'))
            for inst in self:
                if i == inst.verb_head:
                    s += '%-20s ' % inst.verb_stem
                    break
            else:
                s += '%-20s ' % '-'
            for inst in self:
                argstr = '*'
                for (start, end), argid in inst.tagged_spans:
                    if i == start:
                        argstr = f'({argid}{argstr}'
                    if i == end - 1:
                        argstr += ')'
                s += '%-12s ' % argstr
            s += '\n'
        return s

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