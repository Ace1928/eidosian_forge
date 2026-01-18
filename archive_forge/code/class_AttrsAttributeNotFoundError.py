from __future__ import annotations
from typing import ClassVar
class AttrsAttributeNotFoundError(ValueError):
    """
    An *attrs* function couldn't find an attribute that the user asked for.

    .. versionadded:: 16.2.0
    """