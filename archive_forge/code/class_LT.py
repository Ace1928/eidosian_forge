from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class LT(_Operator):
    """ Predicate to test if property values are less than some value.

    Construct and ``LT`` predicate as a dict with ``LT`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size < 10
        dict(size={ LT: 10 })

    """
    pass