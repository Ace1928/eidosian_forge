from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_parallel(trees, notify):
    """ Handle an expression of the form "a, b".

    Parameters
    ----------
    trees : list of lark.tree.Tree
        The children tree for the "parallel" rule. It should always
        contain exactly two items.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression : ObserverExpression
    """
    left, right = trees
    return _handle_tree(left, notify) | _handle_tree(right, notify)