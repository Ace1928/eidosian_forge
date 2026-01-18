import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def _get_parent_indices(self):
    return [(parent, index) for parent in self._parents for index, child in enumerate(parent) if child is self]