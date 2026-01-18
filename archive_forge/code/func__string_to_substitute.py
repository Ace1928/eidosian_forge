import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _string_to_substitute(self, mo: Match, methods_dict: Dict[str, Callable]) -> str:
    """
        Return the string to be substituted for the match.
        """
    matched_text, f_name = mo.groups()
    if f_name not in methods_dict:
        return matched_text
    a_tree = ast.parse(matched_text)
    args_list = [self._get_value_from_ast(a) for a in a_tree.body[0].value.args]
    kwargs_list = {kw.arg: self._get_value_from_ast(kw.value) for kw in a_tree.body[0].value.keywords}
    return methods_dict[f_name](*args_list, **kwargs_list)