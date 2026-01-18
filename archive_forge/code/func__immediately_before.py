import functools
import re
import nltk.tree
def _immediately_before(node):
    """
    Returns the set of all nodes that are immediately before the given
    node.

    Tree node A immediately precedes node B if the last terminal
    symbol (word) produced by A immediately precedes the first
    terminal symbol produced by B.
    """
    try:
        pos = node.treeposition()
        tree = node.root()
    except AttributeError:
        return []
    idx = len(pos) - 1
    while 0 <= idx and pos[idx] == 0:
        idx -= 1
    if idx < 0:
        return []
    pos = list(pos[:idx + 1])
    pos[-1] -= 1
    before = tree[pos]
    return [before] + _rightmost_descendants(before)