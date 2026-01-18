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
def _run_aggregation_cmd(self, session: Optional[ClientSession], explicit_session: bool) -> CommandCursor:
    """Run the full aggregation pipeline for this ChangeStream and return
        the corresponding CommandCursor.
        """
    cmd = self._aggregation_command_class(self._target, CommandCursor, self._aggregation_pipeline(), self._command_options(), explicit_session, result_processor=self._process_result, comment=self._comment)
    return self._client._retryable_read(cmd.get_cursor, self._target._read_preference_for(session), session)