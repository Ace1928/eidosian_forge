from inspect import isclass, signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.decorators import undoc
def _eval_return_type(func: Callable, node: ast.Call, context: EvaluationContext):
    """Evaluate return type of a given callable function.

    Returns the built-in type, a duck or NOT_EVALUATED sentinel.
    """
    try:
        sig = signature(func)
    except ValueError:
        sig = UNKNOWN_SIGNATURE
    not_empty = sig.return_annotation is not Signature.empty
    if not_empty:
        return _resolve_annotation(sig.return_annotation, sig, func, node, context)
    return NOT_EVALUATED