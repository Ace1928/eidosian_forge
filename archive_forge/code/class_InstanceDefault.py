from __future__ import annotations
import logging # isort:skip
import types
from importlib import import_module
from typing import (
from ..has_props import HasProps
from ..serialization import Serializable
from ._sphinx import model_link, property_link, register_type_link
from .bases import Init, Property
from .singletons import Undefined
class InstanceDefault(Generic[I]):
    """ Provide a deferred initializer for Instance defaults.

    This is useful for Bokeh models with Instance properties that should have
    unique default values for every model instance. Using an InstanceDefault
    will afford better user-facing documentation than a lambda initializer.

    """

    def __init__(self, model: type[I], **kwargs: Any) -> None:
        self._model = model
        self._kwargs = kwargs

    def __call__(self) -> I:
        return self._model(**self._kwargs)

    def __repr__(self) -> str:
        kwargs = ', '.join((f'{key}={val}' for key, val in self._kwargs.items()))
        return f'<Instance: {self._model.__qualified_model__}({kwargs})>'