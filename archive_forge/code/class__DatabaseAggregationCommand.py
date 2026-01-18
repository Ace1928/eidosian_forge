from __future__ import annotations
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional, Union
from bson.son import SON
from pymongo import common
from pymongo.collation import validate_collation_or_none
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref
class _DatabaseAggregationCommand(_AggregationCommand):
    _target: Database

    @property
    def _aggregation_target(self) -> int:
        return 1

    @property
    def _cursor_namespace(self) -> str:
        return f'{self._target.name}.$cmd.aggregate'

    @property
    def _database(self) -> Database:
        return self._target

    def _cursor_collection(self, cursor: Mapping[str, Any]) -> Collection:
        """The Collection used for the aggregate command cursor."""
        _, collname = cursor.get('ns', self._cursor_namespace).split('.', 1)
        return self._database[collname]