import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
def _issubtype_with_constraints(variant, constraints, recursive=True):
    """
    Check if the variant is a subtype of either one from constraints.

    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    if variant in constraints:
        return True
    vs = _decompose_type(variant, to_list=False)
    if vs is not None:
        return all((_issubtype_with_constraints(v, constraints, recursive) for v in vs))
    if hasattr(variant, '__origin__') and variant.__origin__ is not None:
        v_origin = variant.__origin__
        v_args = getattr(variant, '__args__', None)
    else:
        v_origin = variant
        v_args = None
    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)
        if cs is not None:
            if _issubtype_with_constraints(variant, cs, recursive):
                return True
        elif hasattr(constraint, '__origin__') and constraint.__origin__ is not None:
            c_origin = constraint.__origin__
            if v_origin == c_origin:
                if not recursive:
                    return True
                c_args = getattr(constraint, '__args__', None)
                if c_args is None or len(c_args) == 0:
                    return True
                if v_args is not None and len(v_args) == len(c_args) and all((issubtype(v_arg, c_arg) for v_arg, c_arg in zip(v_args, c_args))):
                    return True
        elif v_origin == constraint:
            return True
    return False