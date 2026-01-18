import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def _setparent(self, child, index, dry_run=False):
    if not isinstance(child, MultiParentedTree):
        raise TypeError('Can not insert a non-MultiParentedTree into a MultiParentedTree')
    if not dry_run:
        for parent in child._parents:
            if parent is self:
                break
        else:
            child._parents.append(self)