from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional, Union
from bson import _decode_all_selective
from pymongo.errors import NotPrimaryError, OperationFailure
from pymongo.helpers import _check_command_response, _handle_reauth
from pymongo.message import _convert_exception, _GetMore, _OpMsg, _Query
from pymongo.response import PinnedResponse, Response
def _split_message(self, message: Union[tuple[int, Any], tuple[int, Any, int]]) -> tuple[int, Any, int]:
    """Return request_id, data, max_doc_size.

        :Parameters:
          - `message`: (request_id, data, max_doc_size) or (request_id, data)
        """
    if len(message) == 3:
        return message
    else:
        request_id, data = message
        return (request_id, data, 0)