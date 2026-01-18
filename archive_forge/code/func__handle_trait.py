from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_trait(trees, notify):
    """ Handle an element for a named trait.

    Parameters
    ----------
    trees : list of lark.tree.Tree
        The children tree for the "trait" rule.
        It contains only one item.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression : ObserverExpression
    """
    token, = trees
    name = token.value
    return expression_module.trait(name, notify=notify)