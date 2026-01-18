from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class ListPartitionCursorsRequest(proto.Message):
    """Request for ListPartitionCursors.

    Attributes:
        parent (str):
            Required. The subscription for which to retrieve cursors.
            Structured like
            ``projects/{project_number}/locations/{location}/subscriptions/{subscription_id}``.
        page_size (int):
            The maximum number of cursors to return. The
            service may return fewer than this value.
            If unset or zero, all cursors for the parent
            will be returned.
        page_token (str):
            A page token, received from a previous
            ``ListPartitionCursors`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListPartitionCursors`` must match the call that provided
            the page token.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)