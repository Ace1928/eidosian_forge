from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class IN(_Operator):
    """ Predicate to test if property values are in some collection.

    Construct and ``IN`` predicate as a dict with ``IN`` as the key,
    and a list of values to check against.

    .. code-block:: python

        # matches any models with .name in ['a', 'mycircle', 'myline']
        dict(name={ IN: ['a', 'mycircle', 'myline'] })

    """
    pass