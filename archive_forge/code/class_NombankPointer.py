from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class NombankPointer:
    """
    A pointer used by nombank to identify one or more constituents in
    a parse tree.  ``NombankPointer`` is an abstract base class with
    three concrete subclasses:

    - ``NombankTreePointer`` is used to point to single constituents.
    - ``NombankSplitTreePointer`` is used to point to 'split'
      constituents, which consist of a sequence of two or more
      ``NombankTreePointer`` pointers.
    - ``NombankChainTreePointer`` is used to point to entire trace
      chains in a tree.  It consists of a sequence of pieces, which
      can be ``NombankTreePointer`` or ``NombankSplitTreePointer`` pointers.
    """

    def __init__(self):
        if self.__class__ == NombankPointer:
            raise NotImplementedError()