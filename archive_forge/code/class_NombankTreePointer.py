from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
@total_ordering
class NombankTreePointer(NombankPointer):
    """
    wordnum:height*wordnum:height*...
    wordnum:height,

    """

    def __init__(self, wordnum, height):
        self.wordnum = wordnum
        self.height = height

    @staticmethod
    def parse(s):
        pieces = s.split('*')
        if len(pieces) > 1:
            return NombankChainTreePointer([NombankTreePointer.parse(elt) for elt in pieces])
        pieces = s.split(',')
        if len(pieces) > 1:
            return NombankSplitTreePointer([NombankTreePointer.parse(elt) for elt in pieces])
        pieces = s.split(':')
        if len(pieces) != 2:
            raise ValueError('bad nombank pointer %r' % s)
        return NombankTreePointer(int(pieces[0]), int(pieces[1]))

    def __str__(self):
        return f'{self.wordnum}:{self.height}'

    def __repr__(self):
        return 'NombankTreePointer(%d, %d)' % (self.wordnum, self.height)

    def __eq__(self, other):
        while isinstance(other, (NombankChainTreePointer, NombankSplitTreePointer)):
            other = other.pieces[0]
        if not isinstance(other, NombankTreePointer):
            return self is other
        return self.wordnum == other.wordnum and self.height == other.height

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        while isinstance(other, (NombankChainTreePointer, NombankSplitTreePointer)):
            other = other.pieces[0]
        if not isinstance(other, NombankTreePointer):
            return id(self) < id(other)
        return (self.wordnum, -self.height) < (other.wordnum, -other.height)

    def select(self, tree):
        if tree is None:
            raise ValueError('Parse tree not available')
        return tree[self.treepos(tree)]

    def treepos(self, tree):
        """
        Convert this pointer to a standard 'tree position' pointer,
        given that it points to the given tree.
        """
        if tree is None:
            raise ValueError('Parse tree not available')
        stack = [tree]
        treepos = []
        wordnum = 0
        while True:
            if isinstance(stack[-1], Tree):
                if len(treepos) < len(stack):
                    treepos.append(0)
                else:
                    treepos[-1] += 1
                if treepos[-1] < len(stack[-1]):
                    stack.append(stack[-1][treepos[-1]])
                else:
                    stack.pop()
                    treepos.pop()
            elif wordnum == self.wordnum:
                return tuple(treepos[:len(treepos) - self.height - 1])
            else:
                wordnum += 1
                stack.pop()