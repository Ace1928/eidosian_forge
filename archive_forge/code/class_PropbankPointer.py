import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class PropbankPointer:
    """
    A pointer used by propbank to identify one or more constituents in
    a parse tree.  ``PropbankPointer`` is an abstract base class with
    three concrete subclasses:

      - ``PropbankTreePointer`` is used to point to single constituents.
      - ``PropbankSplitTreePointer`` is used to point to 'split'
        constituents, which consist of a sequence of two or more
        ``PropbankTreePointer`` pointers.
      - ``PropbankChainTreePointer`` is used to point to entire trace
        chains in a tree.  It consists of a sequence of pieces, which
        can be ``PropbankTreePointer`` or ``PropbankSplitTreePointer`` pointers.
    """

    def __init__(self):
        if self.__class__ == PropbankPointer:
            raise NotImplementedError()