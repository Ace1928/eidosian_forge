from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
def _raise_if_unacknowledged(self, property_name: str) -> None:
    """Raise an exception on property access if unacknowledged."""
    if not self.__acknowledged:
        raise InvalidOperation(f'A value for {property_name} is not available when the write is unacknowledged. Check the acknowledged attribute to avoid this error.')