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
def __send_message(self, operation: Union[_Query, _GetMore]) -> None:
    """Send a query or getmore operation and handles the response.

        If operation is ``None`` this is an exhaust cursor, which reads
        the next result batch off the exhaust socket instead of
        sending getMore messages to the server.

        Can raise ConnectionFailure.
        """
    client = self.__collection.database.client
    if client._encrypter and self.__exhaust:
        raise InvalidOperation('exhaust cursors do not support auto encryption')
    try:
        response = client._run_operation(operation, self._unpack_response, address=self.__address)
    except OperationFailure as exc:
        if exc.code in _CURSOR_CLOSED_ERRORS or self.__exhaust:
            self.__killed = True
        if exc.timeout:
            self.__die(False)
        else:
            self.close()
        if exc.code in _CURSOR_CLOSED_ERRORS and self.__query_flags & _QUERY_OPTIONS['tailable_cursor']:
            return
        raise
    except ConnectionFailure:
        self.__killed = True
        self.close()
        raise
    except Exception:
        self.close()
        raise
    self.__address = response.address
    if isinstance(response, PinnedResponse):
        if not self.__sock_mgr:
            self.__sock_mgr = _ConnectionManager(response.conn, response.more_to_come)
    cmd_name = operation.name
    docs = response.docs
    if response.from_command:
        if cmd_name != 'explain':
            cursor = docs[0]['cursor']
            self.__id = cursor['id']
            if cmd_name == 'find':
                documents = cursor['firstBatch']
                ns = cursor.get('ns')
                if ns:
                    self.__dbname, self.__collname = ns.split('.', 1)
            else:
                documents = cursor['nextBatch']
            self.__data = deque(documents)
            self.__retrieved += len(documents)
        else:
            self.__id = 0
            self.__data = deque(docs)
            self.__retrieved += len(docs)
    else:
        assert isinstance(response.data, _OpReply)
        self.__id = response.data.cursor_id
        self.__data = deque(docs)
        self.__retrieved += response.data.number_returned
    if self.__id == 0:
        self.close()
    if self.__limit and self.__id and (self.__limit <= self.__retrieved):
        self.close()