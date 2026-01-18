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
class ClusterChangeStream(DatabaseChangeStream[_DocumentType]):
    """A change stream that watches changes on all collections in the cluster.

    Should not be called directly by application developers. Use
    helper method :meth:`pymongo.mongo_client.MongoClient.watch` instead.

    .. versionadded:: 3.7
    """

    def _change_stream_options(self) -> dict[str, Any]:
        options = super()._change_stream_options()
        options['allChangesForCluster'] = True
        return options