from __future__ import annotations
from collections import deque
from typing import (
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
def _try_next(self, get_more_allowed: bool) -> Optional[_DocumentType]:
    """Advance the cursor blocking for at most one getMore command."""
    if not len(self.__data) and (not self.__killed) and get_more_allowed:
        self._refresh()
    if len(self.__data):
        return self.__data.popleft()
    else:
        return None