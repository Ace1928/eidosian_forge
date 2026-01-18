from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class InitialSubscribeResponse(proto.Message):
    """Response to an InitialSubscribeRequest.

    Attributes:
        cursor (google.cloud.pubsublite_v1.types.Cursor):
            The cursor from which the subscriber will
            start receiving messages once flow control
            tokens become available.
    """
    cursor: common.Cursor = proto.Field(proto.MESSAGE, number=1, message=common.Cursor)