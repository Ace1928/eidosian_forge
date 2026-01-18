from __future__ import annotations
import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Callable, Iterable, Iterator, cast
from typing_extensions import Awaitable, AsyncIterable, AsyncIterator, assert_never
import httpx
from ..._utils import is_dict, is_list, consume_sync_iterator, consume_async_iterator
from ..._models import construct_type
from ..._streaming import Stream, AsyncStream
from ...types.beta import AssistantStreamEvent
from ...types.beta.threads import (
from ...types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta
def _emit_sse_event(self, event: AssistantStreamEvent) -> None:
    self._current_event = event
    self.on_event(event)
    self.__current_message_snapshot, new_content = accumulate_event(event=event, current_message_snapshot=self.__current_message_snapshot)
    if self.__current_message_snapshot is not None:
        self.__message_snapshots[self.__current_message_snapshot.id] = self.__current_message_snapshot
    accumulate_run_step(event=event, run_step_snapshots=self.__run_step_snapshots)
    for content_delta in new_content:
        assert self.__current_message_snapshot is not None
        block = self.__current_message_snapshot.content[content_delta.index]
        if block.type == 'text':
            self.on_text_created(block.text)
    if event.event == 'thread.run.completed' or event.event == 'thread.run.cancelled' or event.event == 'thread.run.expired' or (event.event == 'thread.run.failed') or (event.event == 'thread.run.requires_action'):
        self.__current_run = event.data
        if self._current_tool_call:
            self.on_tool_call_done(self._current_tool_call)
    elif event.event == 'thread.run.created' or event.event == 'thread.run.in_progress' or event.event == 'thread.run.cancelling' or (event.event == 'thread.run.queued'):
        self.__current_run = event.data
    elif event.event == 'thread.message.created':
        self.on_message_created(event.data)
    elif event.event == 'thread.message.delta':
        snapshot = self.__current_message_snapshot
        assert snapshot is not None
        message_delta = event.data.delta
        if message_delta.content is not None:
            for content_delta in message_delta.content:
                if content_delta.type == 'text' and content_delta.text:
                    snapshot_content = snapshot.content[content_delta.index]
                    assert snapshot_content.type == 'text'
                    self.on_text_delta(content_delta.text, snapshot_content.text)
                if content_delta.index != self._current_message_content_index:
                    if self._current_message_content is not None:
                        if self._current_message_content.type == 'text':
                            self.on_text_done(self._current_message_content.text)
                        elif self._current_message_content.type == 'image_file':
                            self.on_image_file_done(self._current_message_content.image_file)
                    self._current_message_content_index = content_delta.index
                    self._current_message_content = snapshot.content[content_delta.index]
                self._current_message_content = snapshot.content[content_delta.index]
        self.on_message_delta(event.data.delta, snapshot)
    elif event.event == 'thread.message.completed' or event.event == 'thread.message.incomplete':
        self.__current_message_snapshot = event.data
        self.__message_snapshots[event.data.id] = event.data
        if self._current_message_content_index is not None:
            content = event.data.content[self._current_message_content_index]
            if content.type == 'text':
                self.on_text_done(content.text)
            elif content.type == 'image_file':
                self.on_image_file_done(content.image_file)
        self.on_message_done(event.data)
    elif event.event == 'thread.run.step.created':
        self.__current_run_step_id = event.data.id
        self.on_run_step_created(event.data)
    elif event.event == 'thread.run.step.in_progress':
        self.__current_run_step_id = event.data.id
    elif event.event == 'thread.run.step.delta':
        step_snapshot = self.__run_step_snapshots[event.data.id]
        run_step_delta = event.data.delta
        if run_step_delta.step_details and run_step_delta.step_details.type == 'tool_calls' and (run_step_delta.step_details.tool_calls is not None):
            assert step_snapshot.step_details.type == 'tool_calls'
            for tool_call_delta in run_step_delta.step_details.tool_calls:
                if tool_call_delta.index == self._current_tool_call_index:
                    self.on_tool_call_delta(tool_call_delta, step_snapshot.step_details.tool_calls[tool_call_delta.index])
                if tool_call_delta.index != self._current_tool_call_index:
                    if self._current_tool_call is not None:
                        self.on_tool_call_done(self._current_tool_call)
                    self._current_tool_call_index = tool_call_delta.index
                    self._current_tool_call = step_snapshot.step_details.tool_calls[tool_call_delta.index]
                    self.on_tool_call_created(self._current_tool_call)
                self._current_tool_call = step_snapshot.step_details.tool_calls[tool_call_delta.index]
        self.on_run_step_delta(event.data.delta, step_snapshot)
    elif event.event == 'thread.run.step.completed' or event.event == 'thread.run.step.cancelled' or event.event == 'thread.run.step.expired' or (event.event == 'thread.run.step.failed'):
        if self._current_tool_call:
            self.on_tool_call_done(self._current_tool_call)
        self.on_run_step_done(event.data)
        self.__current_run_step_id = None
    elif event.event == 'thread.created' or event.event == 'thread.message.in_progress' or event.event == 'error':
        ...
    elif TYPE_CHECKING:
        assert_never(event)
    self._current_event = None