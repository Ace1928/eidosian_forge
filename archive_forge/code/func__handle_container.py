import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def _handle_container(self, node: Union[ast.List, ast.Tuple, ast.Set, ast.Dict]) -> Union[List, Tuple, Set, Dict]:
    """Handle container nodes, including List, Set, Tuple and Dict"""
    if isinstance(node, ast.Dict):
        elts = node.keys
        if None in elts:
            raise CannotEval
    else:
        elts = node.elts
    elts = [self[elt] for elt in elts]
    if isinstance(node, ast.List):
        return elts
    if isinstance(node, ast.Tuple):
        return tuple(elts)
    if not all((is_standard_types(elt, check_dict_values=False, deep=True) for elt in elts)):
        raise CannotEval
    if isinstance(node, ast.Set):
        try:
            return set(elts)
        except TypeError:
            raise CannotEval
    assert isinstance(node, ast.Dict)
    pairs = [(elt, self[val]) for elt, val in zip(elts, node.values)]
    try:
        return dict(pairs)
    except TypeError:
        raise CannotEval