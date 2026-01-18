import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
class StreamSynchronizations:

    def __init__(self):
        self.current_sync_states: Dict[StreamId, Dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: Dict[EventId, Dict[StreamId, SeqNum]] = {}
        self.host_sync_state: Dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info('Found Stream with id: %s, but no matching stream creation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?', stream)
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info('Found Event with id: %s, but no matching event creation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?', event)
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info("Found duplicate event creation in the trace for event with id: %s. Assuming the trace for event deletion wasn't caught and backfilling it now. Perhaps the sanitizer was enabled after some torch operations?", event)
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info('Found duplicate Stream creation in the trace for Stream with id: %s. PyTorch Streams are only created once, so this trace entry is ignored.', stream)
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

    def create_event(self, event: EventId) -> None:
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    def delete_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    def record_state(self, event: EventId, stream: StreamId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    def _state_wait_for_other(self, state: Dict[StreamId, SeqNum], other: Dict[StreamId, SeqNum]) -> None:
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_stream_exists(stream)
        self._ensure_event_exists(event)
        self._state_wait_for_other(self.current_sync_states[stream], self.recorded_sync_states[event])

    def all_streams_wait_for_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        for stream in self.current_sync_states.keys():
            self.stream_wait_for_event(stream, event)
        self._state_wait_for_other(self.host_sync_state, self.recorded_sync_states[event])

    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        self._ensure_stream_exists(stream)
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])
        self._state_wait_for_other(self.host_sync_state, self.current_sync_states[stream])

    def sync_all_streams(self) -> None:
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    def is_ordered_after(self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId) -> bool:
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)