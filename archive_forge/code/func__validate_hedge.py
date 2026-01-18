from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def _validate_hedge(hedge: Optional[_Hedge]) -> Optional[_Hedge]:
    """Validate hedge."""
    if hedge is None:
        return None
    if not isinstance(hedge, dict):
        raise TypeError(f'hedge must be a dictionary, not {hedge!r}')
    return hedge