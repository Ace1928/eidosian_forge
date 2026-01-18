import copy
from dataclasses import is_dataclass
from itertools import zip_longest
from typing import TypeVar, Type, Optional, get_type_hints, Mapping, Any
from .config import Config
from .data import Data
from .dataclasses import get_default_value_for_field, create_instance, DefaultValueNotFoundError, get_fields
from .exceptions import (
from .types import (
def _build_value(type_: Type, data: Any, config: Config) -> Any:
    if is_init_var(type_):
        type_ = extract_init_var(type_)
    if is_union(type_):
        return _build_value_for_union(union=type_, data=data, config=config)
    elif is_generic_collection(type_) and is_instance(data, extract_origin_collection(type_)):
        return _build_value_for_collection(collection=type_, data=data, config=config)
    elif is_dataclass(type_) and is_instance(data, Data):
        return from_dict(data_class=type_, data=data, config=config)
    return data