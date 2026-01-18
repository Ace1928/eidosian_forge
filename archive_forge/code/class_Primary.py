from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class Primary(_ServerMode):
    """Primary read preference.

    * When directly connected to one mongod queries are allowed if the server
      is standalone or a replica set primary.
    * When connected to a mongos queries are sent to the primary of a shard.
    * When connected to a replica set queries are sent to the primary of
      the replica set.
    """
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(_PRIMARY)

    def __call__(self, selection: Selection) -> Selection:
        """Apply this read preference to a Selection."""
        return selection.primary_selection

    def __repr__(self) -> str:
        return 'Primary()'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _ServerMode):
            return other.mode == _PRIMARY
        return NotImplemented