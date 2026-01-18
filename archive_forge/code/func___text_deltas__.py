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
def __text_deltas__(self) -> Iterator[str]:
    for event in self:
        if event.event != 'thread.message.delta':
            continue
        for content_delta in event.data.delta.content or []:
            if content_delta.type == 'text' and content_delta.text and content_delta.text.value:
                yield content_delta.text.value