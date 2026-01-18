from __future__ import annotations
from typing import Any
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
class ForwardMsgQueue:
    """Accumulates a session's outgoing ForwardMsgs.

    Each AppSession adds messages to its queue, and the Server periodically
    flushes all session queues and delivers their messages to the appropriate
    clients.

    ForwardMsgQueue is not thread-safe - a queue should only be used from
    a single thread.
    """

    def __init__(self):
        self._queue: list[ForwardMsg] = []
        self._delta_index_map: dict[tuple[int, ...], int] = dict()

    def get_debug(self) -> dict[str, Any]:
        from google.protobuf.json_format import MessageToDict
        return {'queue': [MessageToDict(m) for m in self._queue], 'ids': list(self._delta_index_map.keys())}

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def enqueue(self, msg: ForwardMsg) -> None:
        """Add message into queue, possibly composing it with another message."""
        if not _is_composable_message(msg):
            self._queue.append(msg)
            return
        delta_key = tuple(msg.metadata.delta_path)
        if delta_key in self._delta_index_map:
            index = self._delta_index_map[delta_key]
            old_msg = self._queue[index]
            composed_delta = _maybe_compose_deltas(old_msg.delta, msg.delta)
            if composed_delta is not None:
                new_msg = ForwardMsg()
                new_msg.delta.CopyFrom(composed_delta)
                new_msg.metadata.CopyFrom(msg.metadata)
                self._queue[index] = new_msg
                return
        self._delta_index_map[delta_key] = len(self._queue)
        self._queue.append(msg)

    def clear(self) -> None:
        """Clear the queue."""
        self._queue = []
        self._delta_index_map = dict()

    def flush(self) -> list[ForwardMsg]:
        """Clear the queue and return a list of the messages it contained
        before being cleared.
        """
        queue = self._queue
        self.clear()
        return queue

    def __len__(self) -> int:
        return len(self._queue)