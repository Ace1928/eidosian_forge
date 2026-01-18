from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
 Internal helper for dealing with units associated units properties
        when setting values on ``UnitsSpec`` properties.

        When ``value`` is a dict, this function may mutate the value of the
        associated units property.

        Args:
            obj (HasProps) : instance to update units spec property value for
            value (obj) : new value to set for the property

        Returns:
            copy of ``value``, with 'units' key and value removed when
            applicable

        