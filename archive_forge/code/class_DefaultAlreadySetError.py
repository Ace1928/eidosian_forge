from __future__ import annotations
from typing import ClassVar
class DefaultAlreadySetError(RuntimeError):
    """
    A default has been set when defining the field and is attempted to be reset
    using the decorator.

    .. versionadded:: 17.1.0
    """