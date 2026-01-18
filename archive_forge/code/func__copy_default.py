from __future__ import annotations
import logging # isort:skip
from copy import copy
from typing import (
from ...util.dependencies import uses_pandas
from ...util.strings import nice_join
from ..has_props import HasProps
from ._sphinx import property_link, register_type_link, type_link
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
from .singletons import (
@classmethod
def _copy_default(cls, default: Callable[[], T] | T, *, no_eval: bool=False) -> T:
    """ Return a copy of the default, or a new value if the default
        is specified by a function.

        """
    if not callable(default):
        return copy(default)
    else:
        if no_eval:
            return default
        return default()