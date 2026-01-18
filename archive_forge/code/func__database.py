from __future__ import annotations
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional, Union
from bson.son import SON
from pymongo import common
from pymongo.collation import validate_collation_or_none
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref
@property
def _database(self) -> Database:
    return self._target