from __future__ import annotations
from collections import deque
from typing import (
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
def _maybe_pin_connection(self, conn: Connection) -> None:
    client = self.__collection.database.client
    if not client._should_pin_cursor(self.__session):
        return
    if not self.__sock_mgr:
        conn.pin_cursor()
        conn_mgr = _ConnectionManager(conn, False)
        if self.__id == 0:
            conn_mgr.close()
        else:
            self.__sock_mgr = conn_mgr