from __future__ import annotations
from collections import abc
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import ASCENDING, _csot, common, helpers, message
from pymongo.aggregation import (
from pymongo.bulk import _Bulk
from pymongo.change_stream import CollectionChangeStream
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor, RawBatchCommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.cursor import Cursor, RawBatchCursor
from pymongo.errors import (
from pymongo.helpers import _check_write_command_response
from pymongo.message import _UNICODE_REPLACE_CODEC_OPTIONS
from pymongo.operations import (
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.results import (
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
from pymongo.write_concern import WriteConcern
def _update_retryable(self, criteria: Mapping[str, Any], document: Union[Mapping[str, Any], _Pipeline], upsert: bool=False, multi: bool=False, write_concern: Optional[WriteConcern]=None, op_id: Optional[int]=None, ordered: bool=True, bypass_doc_val: Optional[bool]=False, collation: Optional[_CollationIn]=None, array_filters: Optional[Sequence[Mapping[str, Any]]]=None, hint: Optional[_IndexKeyHint]=None, session: Optional[ClientSession]=None, let: Optional[Mapping[str, Any]]=None, comment: Optional[Any]=None) -> Optional[Mapping[str, Any]]:
    """Internal update / replace helper."""

    def _update(session: Optional[ClientSession], conn: Connection, retryable_write: bool) -> Optional[Mapping[str, Any]]:
        return self._update(conn, criteria, document, upsert=upsert, multi=multi, write_concern=write_concern, op_id=op_id, ordered=ordered, bypass_doc_val=bypass_doc_val, collation=collation, array_filters=array_filters, hint=hint, session=session, retryable_write=retryable_write, let=let, comment=comment)
    return self.__database.client._retryable_write((write_concern or self.write_concern).acknowledged and (not multi), _update, session)