from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def _field_list_from_nontuple_sequence_checked(f: Union[Callable, TypeForm[Any]], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    """Tuples are handled differently due to variadics (tuple[T1, T2, T3, ..., Tn]),
    while list[], sequence[], set[], etc only have one typevar."""
    contained_type: Any
    if len(get_args(f)) == 0:
        assert default_instance in MISSING_SINGLETONS, f'{default_instance} {f}'
        return UnsupportedNestedTypeMessage(f'Sequence type {f} needs either an explicit type or a default to infer from.')
    else:
        contained_type, = get_args(f)
    return _try_field_list_from_sequence_inner(contained_type, default_instance)