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
def accumulate_event(*, event: AssistantStreamEvent, current_message_snapshot: Message | None) -> tuple[Message | None, list[MessageContentDelta]]:
    """Returns a tuple of message snapshot and newly created text message deltas"""
    if event.event == 'thread.message.created':
        return (event.data, [])
    new_content: list[MessageContentDelta] = []
    if event.event != 'thread.message.delta':
        return (current_message_snapshot, [])
    if not current_message_snapshot:
        raise RuntimeError('Encountered a message delta with no previous snapshot')
    data = event.data
    if data.delta.content:
        for content_delta in data.delta.content:
            try:
                block = current_message_snapshot.content[content_delta.index]
            except IndexError:
                current_message_snapshot.content.insert(content_delta.index, cast(MessageContent, construct_type(type_=cast(Any, MessageContent), value=content_delta.model_dump(exclude_unset=True))))
                new_content.append(content_delta)
            else:
                merged = accumulate_delta(cast('dict[object, object]', block.model_dump(exclude_unset=True)), cast('dict[object, object]', content_delta.model_dump(exclude_unset=True)))
                current_message_snapshot.content[content_delta.index] = cast(MessageContent, construct_type(type_=cast(Any, MessageContent), value=merged))
    return (current_message_snapshot, new_content)