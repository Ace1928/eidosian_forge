from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
class FlowControlBatcher:
    _client_tokens: _AggregateRequest
    _pending_tokens: _AggregateRequest

    def __init__(self):
        self._client_tokens = _AggregateRequest()
        self._pending_tokens = _AggregateRequest()

    def add(self, request: FlowControlRequest):
        self._client_tokens += request
        self._pending_tokens += request

    def on_messages(self, messages: List[SequencedMessage]):
        byte_size = 0
        for message in messages:
            byte_size += message.size_bytes
        self._client_tokens += FlowControlRequest(allowed_bytes=-byte_size, allowed_messages=-len(messages))

    def request_for_restart(self) -> Optional[FlowControlRequest]:
        self._pending_tokens = _AggregateRequest()
        return self._client_tokens.to_optional()

    def release_pending_request(self) -> Optional[FlowControlRequest]:
        request = self._pending_tokens
        self._pending_tokens = _AggregateRequest()
        return request.to_optional()