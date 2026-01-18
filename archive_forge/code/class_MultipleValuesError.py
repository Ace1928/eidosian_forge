from __future__ import annotations
from typing import (
class MultipleValuesError(LookupError):
    """
    Exception raised when :class:`Headers` has more than one value for a key.

    """

    def __str__(self) -> str:
        if len(self.args) == 1:
            return repr(self.args[0])
        return super().__str__()