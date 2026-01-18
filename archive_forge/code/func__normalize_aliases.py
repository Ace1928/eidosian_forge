import collections.abc
import io
import itertools
import types
import typing
def _normalize_aliases(type_: Type) -> Type:
    if isinstance(type_, typing.TypeVar):
        return type_
    assert _hashable(type_), '_normalize_aliases should only be called on element types'
    if type_ in BUILTINS_MAPPING:
        return BUILTINS_MAPPING[type_]
    return type_