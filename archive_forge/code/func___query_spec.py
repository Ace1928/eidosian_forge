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
def __query_spec(self) -> Mapping[str, Any]:
    """Get the spec to use for a query."""
    operators: dict[str, Any] = {}
    if self.__ordering:
        operators['$orderby'] = self.__ordering
    if self.__explain:
        operators['$explain'] = True
    if self.__hint:
        operators['$hint'] = self.__hint
    if self.__let:
        operators['let'] = self.__let
    if self.__comment:
        operators['$comment'] = self.__comment
    if self.__max_scan:
        operators['$maxScan'] = self.__max_scan
    if self.__max_time_ms is not None:
        operators['$maxTimeMS'] = self.__max_time_ms
    if self.__max:
        operators['$max'] = self.__max
    if self.__min:
        operators['$min'] = self.__min
    if self.__return_key is not None:
        operators['$returnKey'] = self.__return_key
    if self.__show_record_id is not None:
        operators['$showDiskLoc'] = self.__show_record_id
    if self.__snapshot is not None:
        operators['$snapshot'] = self.__snapshot
    if operators:
        spec = copy.copy(self.__spec)
        if '$query' not in spec:
            spec = SON([('$query', spec)])
        if not isinstance(spec, SON):
            spec = SON(spec)
        spec.update(operators)
        return spec
    elif 'query' in self.__spec and (len(self.__spec) == 1 or next(iter(self.__spec)) == 'query'):
        return SON({'$query': self.__spec})
    return self.__spec