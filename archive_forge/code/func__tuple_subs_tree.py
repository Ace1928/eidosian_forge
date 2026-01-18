import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _tuple_subs_tree(tp, tvars=None, args=None):
    """ ad-hoc function (inspired by union) for legacy typing """
    if tp is Tuple:
        return Tuple
    tree_args = _subs_tree(tp, tvars, args)
    return (Tuple,) + tuple(tree_args)