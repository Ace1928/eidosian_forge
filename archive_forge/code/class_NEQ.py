from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class NEQ(_Operator):
    """ Predicate to test if property values are unequal to some value.

    Construct and ``NEQ`` predicate as a dict with ``NEQ`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size != 10
        dict(size={ NEQ: 10 })

    """
    pass