import functools
import re
import nltk.tree
def _unique_descendants(node):
    """
    Returns the list of all nodes descended from the given node, where
    there is only a single path of descent.
    """
    results = []
    current = node
    while current and _istree(current) and (len(current) == 1):
        current = current[0]
        results.append(current)
    return results