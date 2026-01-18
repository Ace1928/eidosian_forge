import functools
import re
import nltk.tree
def _before(node):
    """
    Returns the set of all nodes that are before the given node.
    """
    try:
        pos = node.treeposition()
        tree = node.root()
    except AttributeError:
        return []
    return [tree[x] for x in tree.treepositions() if x[:len(pos)] < pos[:len(x)]]