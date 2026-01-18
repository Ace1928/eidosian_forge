from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_anytrait(trees, notify):
    """ Handle an anytrait element.

    Parameters
    ----------
    trees : list of lark.tree.Tree
        The children tree for the "trait" rule. This should be empty.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression : ObserverExpression
    """
    return expression_module.anytrait(notify=notify)