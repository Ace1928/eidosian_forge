from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class SequencedCommitCursorRequest(proto.Message):
    """Streaming request to update the committed cursor. Subsequent
    SequencedCommitCursorRequests override outstanding ones.

    Attributes:
        cursor (google.cloud.pubsublite_v1.types.Cursor):
            The new value for the committed cursor.
    """
    cursor: common.Cursor = proto.Field(proto.MESSAGE, number=1, message=common.Cursor)