import sys
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, NamedTuple, Type
from .fields import Required
from .main import BaseModel, create_model
from .typing import is_typeddict, is_typeddict_special

    Create a `BaseModel` based on the fields of a named tuple.
    A named tuple can be created with `typing.NamedTuple` and declared annotations
    but also with `collections.namedtuple`, in this case we consider all fields
    to have type `Any`.
    