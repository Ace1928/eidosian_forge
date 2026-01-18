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
def _retryable_read_command(self, command: Union[str, MutableMapping[str, Any]], session: Optional[ClientSession]=None) -> dict[str, Any]:
    """Same as command but used for retryable read commands."""
    read_preference = session and session._txn_read_preference() or ReadPreference.PRIMARY

    def _cmd(session: Optional[ClientSession], _server: Server, conn: Connection, read_preference: _ServerMode) -> dict[str, Any]:
        return self._command(conn, command, read_preference=read_preference, session=session)
    return self.__client._retryable_read(_cmd, read_preference, session)