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
class ContainerProperty(ParameterizedProperty[T]):
    """ A base class for Container-like type properties.

    """

    def _may_have_unstable_default(self) -> bool:
        return self._default is not Undefined