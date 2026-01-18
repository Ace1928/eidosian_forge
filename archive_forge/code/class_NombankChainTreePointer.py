from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class NombankChainTreePointer(NombankPointer):

    def __init__(self, pieces):
        self.pieces = pieces
        'A list of the pieces that make up this chain.  Elements may\n           be either ``NombankSplitTreePointer`` or\n           ``NombankTreePointer`` pointers.'

    def __str__(self):
        return '*'.join(('%s' % p for p in self.pieces))

    def __repr__(self):
        return '<NombankChainTreePointer: %s>' % self

    def select(self, tree):
        if tree is None:
            raise ValueError('Parse tree not available')
        return Tree('*CHAIN*', [p.select(tree) for p in self.pieces])