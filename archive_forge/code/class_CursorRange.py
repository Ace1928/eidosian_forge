from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class CursorRange(proto.Message):
    """Cursors for a subrange of published messages.

        Attributes:
            start_cursor (google.cloud.pubsublite_v1.types.Cursor):
                The cursor of the message at the start index.
                The cursors for remaining messages up to the end
                index (exclusive) are sequential.
            start_index (int):
                Index of the message in the published batch
                that corresponds to the start cursor. Inclusive.
            end_index (int):
                Index of the last message in this range.
                Exclusive.
        """
    start_cursor: common.Cursor = proto.Field(proto.MESSAGE, number=1, message=common.Cursor)
    start_index: int = proto.Field(proto.INT32, number=2)
    end_index: int = proto.Field(proto.INT32, number=3)