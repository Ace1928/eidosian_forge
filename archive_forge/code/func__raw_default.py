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
def _raw_default(self, *, no_eval: bool=False) -> T:
    """ Return the untransformed default value.

        The raw_default() needs to be validated and transformed by
        prepare_value() before use, and may also be replaced later by
        subclass overrides or by themes.

        """
    return self._copy_default(self._default, no_eval=no_eval)