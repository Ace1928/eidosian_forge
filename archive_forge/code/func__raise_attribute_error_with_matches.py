from __future__ import annotations
import logging # isort:skip
import difflib
from typing import (
from weakref import WeakSet
from ..settings import settings
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import (
from .types import ID
def _raise_attribute_error_with_matches(self, name: str, properties: Iterable[str]) -> NoReturn:
    matches, text = (difflib.get_close_matches(name.lower(), properties), 'similar')
    if not matches:
        matches, text = (sorted(properties), 'possible')
    raise AttributeError(f'unexpected attribute {name!r} to {self.__class__.__name__}, {text} attributes are {nice_join(matches)}')