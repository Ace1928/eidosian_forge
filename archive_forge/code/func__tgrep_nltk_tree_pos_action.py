import functools
import re
import nltk.tree
def _tgrep_nltk_tree_pos_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    which returns true if the node is located at a specific tree
    position.
    """
    node_tree_position = tuple((int(x) for x in tokens if x.isdigit()))
    return (lambda i: lambda n, m=None, l=None: hasattr(n, 'treeposition') and n.treeposition() == i)(node_tree_position)