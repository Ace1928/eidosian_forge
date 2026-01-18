from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class SequencedCommitCursorResponse(proto.Message):
    """Response to a SequencedCommitCursorRequest.

    Attributes:
        acknowledged_commits (int):
            The number of outstanding
            SequencedCommitCursorRequests acknowledged by
            this response. Note that
            SequencedCommitCursorRequests are acknowledged
            in the order that they are received.
    """
    acknowledged_commits: int = proto.Field(proto.INT64, number=1)