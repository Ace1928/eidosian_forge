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
class _ModelResolver:
    """ """
    _known_models: dict[str, type[HasProps]]

    def __init__(self) -> None:
        self._known_models = {}

    def add(self, cls: type[HasProps]) -> None:
        if not (issubclass(cls, Local) or cls.__name__.startswith('_')):
            previous = self._known_models.get(cls.__qualified_model__, None)
            if previous is not None and (not hasattr(cls, '__implementation__')):
                raise Warning(f"Duplicate qualified model declaration of '{cls.__qualified_model__}'. Previous definition: {previous}")
            self._known_models[cls.__qualified_model__] = cls

    def remove(self, cls: type[HasProps]) -> None:
        del self._known_models[cls.__qualified_model__]

    @property
    def known_models(self) -> dict[str, type[HasProps]]:
        return dict(self._known_models)

    def clear_extensions(self) -> None:

        def is_extension(obj: type[HasProps]) -> bool:
            return getattr(obj, '__implementation__', None) is not None or getattr(obj, '__javascript__', None) is not None or getattr(obj, '__css__', None) is not None
        self._known_models = {key: val for key, val in self._known_models.items() if not is_extension(val)}