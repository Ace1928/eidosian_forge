import functools
import re
import nltk.tree
def _tgrep_node_label_pred_use_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    which describes the use of a previously bound node label.

    Called for expressions like (`tgrep_node_label_use_pred`)::

        =s

    when they appear inside a tgrep_node_expr (for example, inside a
    relation).  The predicate returns true if and only if its node
    argument is identical the the node looked up in the node label
    dictionary using the node's label.
    """
    assert len(tokens) == 1
    assert tokens[0].startswith('=')
    node_label = tokens[0][1:]

    def node_label_use_pred(n, m=None, l=None):
        if l is None or node_label not in l:
            raise TgrepException(f'node_label ={node_label} not bound in pattern')
        node = l[node_label]
        return n is node
    return node_label_use_pred