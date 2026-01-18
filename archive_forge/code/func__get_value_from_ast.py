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
def _get_value_from_ast(self, obj):
    """
        Return the value of the ast object.
        """
    if isinstance(obj, ast.Num):
        return obj.n
    elif isinstance(obj, ast.Str):
        return obj.s
    elif isinstance(obj, ast.List):
        return [self._get_value_from_ast(e) for e in obj.elts]
    elif isinstance(obj, ast.Tuple):
        return tuple([self._get_value_from_ast(e) for e in obj.elts])
    elif isinstance(obj, ast.NameConstant):
        return obj.value
    raise NameError(f"name '{obj.id}' is not defined")