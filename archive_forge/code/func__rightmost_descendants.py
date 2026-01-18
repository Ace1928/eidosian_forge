import functools
import re
import nltk.tree
def _rightmost_descendants(node):
    """
    Returns the set of all nodes descended in some way through
    right branches from this node.
    """
    try:
        rightmost_leaf = max(node.treepositions())
    except AttributeError:
        return []
    return [node[rightmost_leaf[:i]] for i in range(1, len(rightmost_leaf) + 1)]