from nltk.probability import ProbabilisticMixIn
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.tree import Tree
class ImmutableTree(Tree):

    def __init__(self, node, children=None):
        super().__init__(node, children)
        try:
            self._hash = hash((self._label, tuple(self)))
        except (TypeError, ValueError) as e:
            raise ValueError('%s: node value and children must be immutable' % type(self).__name__) from e

    def __setitem__(self, index, value):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __setslice__(self, i, j, value):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __delitem__(self, index):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __delslice__(self, i, j):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __iadd__(self, other):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __imul__(self, other):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def append(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def extend(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def pop(self, v=None):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def remove(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def reverse(self):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def sort(self):
        raise ValueError('%s may not be modified' % type(self).__name__)

    def __hash__(self):
        return self._hash

    def set_label(self, value):
        """
        Set the node label.  This will only succeed the first time the
        node label is set, which should occur in ImmutableTree.__init__().
        """
        if hasattr(self, '_label'):
            raise ValueError('%s may not be modified' % type(self).__name__)
        self._label = value