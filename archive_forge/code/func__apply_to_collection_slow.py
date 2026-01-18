import dataclasses
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
def _apply_to_collection_slow(data: Any, dtype: Union[type, Any, Tuple[Union[type, Any]]], function: Callable, *args: Any, wrong_dtype: Optional[Union[type, Tuple[type, ...]]]=None, include_none: bool=True, allow_frozen: bool=False, **kwargs: Any) -> Any:
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)
    elem_type = type(data)
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = _apply_to_collection_slow(v, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, allow_frozen=allow_frozen, **kwargs)
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))
    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and (not isinstance(data, str))
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = _apply_to_collection_slow(d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, allow_frozen=allow_frozen, **kwargs)
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)
    if is_dataclass_instance(data):
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = _apply_to_collection_slow(field_value, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, allow_frozen=allow_frozen, **kwargs)
            if not field_init or (not include_none and v is None):
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                if allow_frozen:
                    break
                raise ValueError('A frozen dataclass was passed to `apply_to_collection` but this is not allowed.') from e
        return result
    return data