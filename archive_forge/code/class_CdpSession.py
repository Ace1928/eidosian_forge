import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
class CdpSession(CdpBase):
    """Contains the state for a CDP session.

    Generally you should not instantiate this object yourself; you should call
    :meth:`CdpConnection.open_session`.
    """

    def __init__(self, ws, session_id, target_id):
        """Constructor.

        :param trio_websocket.WebSocketConnection ws:
        :param devtools.target.SessionID session_id:
        :param devtools.target.TargetID target_id:
        """
        super().__init__(ws, session_id, target_id)
        self._dom_enable_count = 0
        self._dom_enable_lock = trio.Lock()
        self._page_enable_count = 0
        self._page_enable_lock = trio.Lock()

    @asynccontextmanager
    async def dom_enable(self):
        """A context manager that executes ``dom.enable()`` when it enters and
        then calls ``dom.disable()``.

        This keeps track of concurrent callers and only disables DOM
        events when all callers have exited.
        """
        global devtools
        async with self._dom_enable_lock:
            self._dom_enable_count += 1
            if self._dom_enable_count == 1:
                await self.execute(devtools.dom.enable())
        yield
        async with self._dom_enable_lock:
            self._dom_enable_count -= 1
            if self._dom_enable_count == 0:
                await self.execute(devtools.dom.disable())

    @asynccontextmanager
    async def page_enable(self):
        """A context manager that executes ``page.enable()`` when it enters and
        then calls ``page.disable()`` when it exits.

        This keeps track of concurrent callers and only disables page
        events when all callers have exited.
        """
        global devtools
        async with self._page_enable_lock:
            self._page_enable_count += 1
            if self._page_enable_count == 1:
                await self.execute(devtools.page.enable())
        yield
        async with self._page_enable_lock:
            self._page_enable_count -= 1
            if self._page_enable_count == 0:
                await self.execute(devtools.page.disable())