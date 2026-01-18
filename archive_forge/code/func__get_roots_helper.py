import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def _get_roots_helper(self, result):
    if self._parents:
        for parent in self._parents:
            parent._get_roots_helper(result)
    else:
        result[id(self)] = self
    return result