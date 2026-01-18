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
def _generators(class_dict: dict[str, Any]):
    generators: dict[str, PropertyDescriptorFactory[Any]] = {}
    for name, generator in tuple(class_dict.items()):
        if isinstance(generator, PropertyDescriptorFactory):
            del class_dict[name]
            generators[name] = generator
    return generators