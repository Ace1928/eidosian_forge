import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def _frozen_class(self):
    from nltk.tree.immutable import ImmutableMultiParentedTree
    return ImmutableMultiParentedTree