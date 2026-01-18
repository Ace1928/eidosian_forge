from __future__ import annotations
from collections import deque
from typing import (
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
@property
def _post_batch_resume_token(self) -> Optional[Mapping[str, Any]]:
    """Retrieve the postBatchResumeToken from the response to a
        changeStream aggregate or getMore.
        """
    return self.__postbatchresumetoken