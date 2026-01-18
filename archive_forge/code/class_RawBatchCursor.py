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
class RawBatchCursor(Cursor, Generic[_DocumentType]):
    """A cursor / iterator over raw batches of BSON data from a query result."""
    _query_class = _RawBatchQuery
    _getmore_class = _RawBatchGetMore

    def __init__(self, collection: Collection[_DocumentType], *args: Any, **kwargs: Any) -> None:
        """Create a new cursor / iterator over raw batches of BSON data.

        Should not be called directly by application developers -
        see :meth:`~pymongo.collection.Collection.find_raw_batches`
        instead.

        .. seealso:: The MongoDB documentation on `cursors <https://dochub.mongodb.org/core/cursors>`_.
        """
        super().__init__(collection, *args, **kwargs)

    def _unpack_response(self, response: Union[_OpReply, _OpMsg], cursor_id: Optional[int], codec_options: CodecOptions[Mapping[str, Any]], user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[_DocumentOut]:
        raw_response = response.raw_response(cursor_id, user_fields=user_fields)
        if not legacy_response:
            _convert_raw_document_lists_to_streams(raw_response[0])
        return cast(List['_DocumentOut'], raw_response)

    def explain(self) -> _DocumentType:
        """Returns an explain plan record for this cursor.

        .. seealso:: The MongoDB documentation on `explain <https://dochub.mongodb.org/core/explain>`_.
        """
        clone = self._clone(deepcopy=True, base=Cursor(self.collection))
        return clone.explain()

    def __getitem__(self, index: Any) -> NoReturn:
        raise InvalidOperation('Cannot call __getitem__ on RawBatchCursor')