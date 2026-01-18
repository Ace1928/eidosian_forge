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
def _insert_one(self, doc: Mapping[str, Any], ordered: bool, write_concern: WriteConcern, op_id: Optional[int], bypass_doc_val: bool, session: Optional[ClientSession], comment: Optional[Any]=None) -> Any:
    """Internal helper for inserting a single document."""
    write_concern = write_concern or self.write_concern
    acknowledged = write_concern.acknowledged
    command = SON([('insert', self.name), ('ordered', ordered), ('documents', [doc])])
    if comment is not None:
        command['comment'] = comment

    def _insert_command(session: Optional[ClientSession], conn: Connection, retryable_write: bool) -> None:
        if bypass_doc_val:
            command['bypassDocumentValidation'] = True
        result = conn.command(self.__database.name, command, write_concern=write_concern, codec_options=self.__write_response_codec_options, session=session, client=self.__database.client, retryable_write=retryable_write)
        _check_write_command_response(result)
    self.__database.client._retryable_write(acknowledged, _insert_command, session)
    if not isinstance(doc, RawBSONDocument):
        return doc.get('_id')
    return None