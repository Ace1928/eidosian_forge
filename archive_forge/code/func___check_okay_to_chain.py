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
def __check_okay_to_chain(self) -> None:
    """Check if it is okay to chain more options onto this cursor."""
    if self.__retrieved or self.__id is not None:
        raise InvalidOperation('cannot set options after executing query')