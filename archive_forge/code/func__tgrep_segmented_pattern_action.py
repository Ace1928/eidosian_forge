import functools
import re
import nltk.tree
def _tgrep_segmented_pattern_action(_s, _l, tokens):
    """
    Builds a lambda function representing a segmented pattern.

    Called for expressions like (`tgrep_expr_labeled`)::

        =s .. =v < =n

    This is a segmented pattern, a tgrep2 expression which begins with
    a node label.

    The problem is that for segemented_pattern_action (': =v < =s'),
    the first element (in this case, =v) is specifically selected by
    virtue of matching a particular node in the tree; to retrieve
    the node, we need the label, not a lambda function.  For node
    labels inside a tgrep_node_expr, we need a lambda function which
    returns true if the node visited is the same as =v.

    We solve this by creating two copies of a node_label_use in the
    grammar; the label use inside a tgrep_expr_labeled has a separate
    parse action to the pred use inside a node_expr.  See
    `_tgrep_node_label_use_action` and
    `_tgrep_node_label_pred_use_action`.
    """
    node_label = tokens[0]
    reln_preds = tokens[1:]

    def pattern_segment_pred(n, m=None, l=None):
        """This predicate function ignores its node argument."""
        if l is None or node_label not in l:
            raise TgrepException(f'node_label ={node_label} not bound in pattern')
        node = l[node_label]
        return all((pred(node, m, l) for pred in reln_preds))
    return pattern_segment_pred