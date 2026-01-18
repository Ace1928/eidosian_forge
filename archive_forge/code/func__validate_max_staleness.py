from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def _validate_max_staleness(max_staleness: Any) -> int:
    """Validate max_staleness."""
    if max_staleness == -1:
        return -1
    if not isinstance(max_staleness, int):
        raise TypeError(_invalid_max_staleness_msg(max_staleness))
    if max_staleness <= 0:
        raise ValueError(_invalid_max_staleness_msg(max_staleness))
    return max_staleness