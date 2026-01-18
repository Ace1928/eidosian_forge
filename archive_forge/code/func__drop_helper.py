from __future__ import annotations
from copy import deepcopy
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.dbref import DBRef
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _DatabaseAggregationCommand
from pymongo.change_stream import DatabaseChangeStream
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.errors import CollectionInvalid, InvalidName, InvalidOperation
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
def _drop_helper(self, name: str, session: Optional[ClientSession]=None, comment: Optional[Any]=None) -> dict[str, Any]:
    command = SON([('drop', name)])
    if comment is not None:
        command['comment'] = comment
    with self.__client._conn_for_writes(session) as connection:
        return self._command(connection, command, allowable_errors=['ns not found', 26], write_concern=self._write_concern_for(session), parse_write_concern_error=True, session=session)