from __future__ import annotations
from collections import deque
from typing import (
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
def __end_session(self, synchronous: bool) -> None:
    if self.__session and (not self.__explicit_session):
        self.__session._end_session(lock=synchronous)
        self.__session = None