from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence
class _Entry:
    """
    A helper for SpatialDict.

    The implementation of SpatialDict has the same instance of _Entry
    stored for multiple keys so that updating the value for all keys
    can be done by assigning the new value to _Entry.value only once.
    """

    def __init__(self, value):
        self.value = value