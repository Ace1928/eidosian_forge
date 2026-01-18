from __future__ import annotations
import copy
import warnings
from collections import deque
from typing import (
from bson import RE_TYPE, _convert_raw_document_lists_to_streams
from bson.code import Code
from bson.son import SON
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import (
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.lock import _create_lock
from pymongo.message import (
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _CollationIn, _DocumentOut, _DocumentType
def __die(self, synchronous: bool=False) -> None:
    """Closes this cursor."""
    try:
        already_killed = self.__killed
    except AttributeError:
        return
    self.__killed = True
    if self.__id and (not already_killed):
        cursor_id = self.__id
        assert self.__address is not None
        address = _CursorAddress(self.__address, f'{self.__dbname}.{self.__collname}')
    else:
        cursor_id = 0
        address = None
    self.__collection.database.client._cleanup_cursor(synchronous, cursor_id, address, self.__sock_mgr, self.__session, self.__explicit_session)
    if not self.__explicit_session:
        self.__session = None
    self.__sock_mgr = None