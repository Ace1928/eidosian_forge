from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class GT(_Operator):
    """ Predicate to test if property values are greater than some value.

    Construct and ``GT`` predicate as a dict with ``GT`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size > 10
        dict(size={ GT: 10 })

    """
    pass