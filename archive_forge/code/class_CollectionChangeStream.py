from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Type, Union
from bson import CodecOptions, _bson_to_dict
from bson.raw_bson import RawBSONDocument
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import (
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class CollectionChangeStream(ChangeStream[_DocumentType]):
    """A change stream that watches changes on a single collection.

    Should not be called directly by application developers. Use
    helper method :meth:`pymongo.collection.Collection.watch` instead.

    .. versionadded:: 3.7
    """
    _target: Collection[_DocumentType]

    @property
    def _aggregation_command_class(self) -> Type[_CollectionAggregationCommand]:
        return _CollectionAggregationCommand

    @property
    def _client(self) -> MongoClient[_DocumentType]:
        return self._target.database.client