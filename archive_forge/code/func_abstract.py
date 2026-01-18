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
def abstract(cls: C) -> C:
    """ A decorator to mark abstract base classes derived from |HasProps|.

    """
    if not issubclass(cls, HasProps):
        raise TypeError(f'{cls.__name__} is not a subclass of HasProps')
    _abstract_classes.add(cls)
    cls.__doc__ = append_docstring(cls.__doc__, _ABSTRACT_ADMONITION)
    return cls