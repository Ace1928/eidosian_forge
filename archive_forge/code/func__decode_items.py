import copy
import json
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import (MISSING,
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (Any, Collection, Mapping, Union, get_type_hints,
from uuid import UUID
from typing_inspect import is_union_type  # type: ignore
from dataclasses_json import cfg
from dataclasses_json.utils import (_get_type_cons, _get_type_origin,
def _decode_items(type_args, xs, infer_missing):
    """
    This is a tricky situation where we need to check both the annotated
    type info (which is usually a type from `typing`) and check the
    value's type directly using `type()`.

    If the type_arg is a generic we can use the annotated type, but if the
    type_arg is a typevar we need to extract the reified type information
    hence the check of `is_dataclass(vs)`
    """

    def _decode_item(type_arg, x):
        if is_dataclass(type_arg) or is_dataclass(xs):
            return _decode_dataclass(type_arg, x, infer_missing)
        if _is_supported_generic(type_arg):
            return _decode_generic(type_arg, x, infer_missing)
        return x

    def handle_pep0673(pre_0673_hint: str) -> Union[Type, str]:
        for module in sys.modules:
            maybe_resolved = getattr(sys.modules[module], type_args, None)
            if maybe_resolved:
                return maybe_resolved
        warnings.warn(f'Could not resolve self-reference for type {pre_0673_hint}, decoded type might be incorrect or decode might fail altogether.')
        return pre_0673_hint
    if sys.version_info.minor < 11 and type_args is not type and (type(type_args) is str):
        type_args = handle_pep0673(type_args)
    if _isinstance_safe(type_args, Collection) and (not _issubclass_safe(type_args, Enum)):
        if len(type_args) == len(xs):
            return list((_decode_item(type_arg, x) for type_arg, x in zip(type_args, xs)))
        else:
            raise TypeError(f'Number of types specified in the collection type {str(type_args)} does not match number of elements in the collection. In case you are working with tuplestake a look at this document docs.python.org/3/library/typing.html#annotating-tuples.')
    return list((_decode_item(type_args, x) for x in xs))