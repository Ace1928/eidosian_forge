import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def _delparent(self, child, index):
    assert isinstance(child, MultiParentedTree)
    assert self[index] is child
    assert len([p for p in child._parents if p is self]) == 1
    for i, c in enumerate(self):
        if c is child and i != index:
            break
    else:
        child._parents.remove(self)