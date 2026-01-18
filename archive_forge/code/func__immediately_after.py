import functools
import re
import nltk.tree
def _immediately_after(node):
    """
    Returns the set of all nodes that are immediately after the given
    node.

    Tree node A immediately follows node B if the first terminal
    symbol (word) produced by A immediately follows the last
    terminal symbol produced by B.
    """
    try:
        pos = node.treeposition()
        tree = node.root()
        current = node.parent()
    except AttributeError:
        return []
    idx = len(pos) - 1
    while 0 <= idx and pos[idx] == len(current) - 1:
        idx -= 1
        current = current.parent()
    if idx < 0:
        return []
    pos = list(pos[:idx + 1])
    pos[-1] += 1
    after = tree[pos]
    return [after] + _leftmost_descendants(after)