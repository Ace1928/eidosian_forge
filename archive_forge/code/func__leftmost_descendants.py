import functools
import re
import nltk.tree
def _leftmost_descendants(node):
    """
    Returns the set of all nodes descended in some way through
    left branches from this node.
    """
    try:
        treepos = node.treepositions()
    except AttributeError:
        return []
    return [node[x] for x in treepos[1:] if all((y == 0 for y in x))]