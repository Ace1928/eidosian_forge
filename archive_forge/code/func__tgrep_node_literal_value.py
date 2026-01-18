import functools
import re
import nltk.tree
def _tgrep_node_literal_value(node):
    """
    Gets the string value of a given parse tree node, for comparison
    using the tgrep node literal predicates.
    """
    return node.label() if _istree(node) else str(node)