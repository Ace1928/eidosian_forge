from __future__ import annotations
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional, Union
from bson.son import SON
from pymongo import common
from pymongo.collation import validate_collation_or_none
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref
class _CollectionAggregationCommand(_AggregationCommand):
    _target: Collection

    @property
    def _aggregation_target(self) -> str:
        return self._target.name

    @property
    def _cursor_namespace(self) -> str:
        return self._target.full_name

    def _cursor_collection(self, cursor: Mapping[str, Any]) -> Collection:
        """The Collection used for the aggregate command cursor."""
        return self._target

    @property
    def _database(self) -> Database:
        return self._target.database