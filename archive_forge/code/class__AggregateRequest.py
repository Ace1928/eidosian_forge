from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
class _AggregateRequest:
    _request: FlowControlRequest.meta.pb

    def __init__(self):
        self._request = FlowControlRequest.meta.pb()

    def __add__(self, other: FlowControlRequest):
        other_pb = other._pb
        self._request.allowed_bytes = self._request.allowed_bytes + other_pb.allowed_bytes
        self._request.allowed_bytes = min(self._request.allowed_bytes, _MAX_INT64)
        self._request.allowed_messages = self._request.allowed_messages + other_pb.allowed_messages
        self._request.allowed_messages = min(self._request.allowed_messages, _MAX_INT64)
        return self

    def to_optional(self) -> Optional[FlowControlRequest]:
        allowed_messages = _clamp(self._request.allowed_messages)
        allowed_bytes = _clamp(self._request.allowed_bytes)
        if allowed_messages == 0 and allowed_bytes == 0:
            return None
        request = FlowControlRequest()
        request._pb.allowed_messages = allowed_messages
        request._pb.allowed_bytes = allowed_bytes
        return request