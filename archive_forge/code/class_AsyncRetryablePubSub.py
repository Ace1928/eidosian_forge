from __future__ import annotations
import typing
import logging
import datetime
from redis.commands import (
from redis.client import (
from redis.asyncio.client import (
from aiokeydb.v2.connection import (
from typing import Any, Iterable, Mapping, Callable, Union, overload, TYPE_CHECKING
from typing_extensions import Literal
from .utils.helpers import get_retryable_wrapper
class AsyncRetryablePubSub(AsyncPubSub):
    """
    Retryable PubSub
    """

    @retryable_wrapper
    async def subscribe(self, *args: 'ChannelT', **kwargs: typing.Callable):
        """
        Subscribe to channels. Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """
        return await super().subscribe(*args, **kwargs)

    @retryable_wrapper
    def unsubscribe(self, *args) -> typing.Awaitable:
        """
        Unsubscribe from the supplied channels. If empty, unsubscribe from
        all channels
        """
        return super().unsubscribe(*args)

    @retryable_wrapper
    async def listen(self) -> typing.AsyncIterator:
        """Listen for messages on channels this client has been subscribed to"""
        async for response in super().listen():
            yield response