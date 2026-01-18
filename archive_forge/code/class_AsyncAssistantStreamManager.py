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
class AsyncAssistantStreamManager(Generic[AsyncAssistantEventHandlerT]):
    """Wrapper over AsyncAssistantStreamEventHandler that is returned by `.stream()`
    so that an async context manager can be used without `await`ing the
    original client call.

    ```py
    async with client.threads.create_and_run_stream(...) as stream:
        async for event in stream:
            ...
    ```
    """

    def __init__(self, api_request: Awaitable[AsyncStream[AssistantStreamEvent]], *, event_handler: AsyncAssistantEventHandlerT) -> None:
        self.__stream: AsyncStream[AssistantStreamEvent] | None = None
        self.__event_handler = event_handler
        self.__api_request = api_request

    async def __aenter__(self) -> AsyncAssistantEventHandlerT:
        self.__stream = await self.__api_request
        self.__event_handler._init(self.__stream)
        return self.__event_handler

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self.__stream is not None:
            await self.__stream.close()