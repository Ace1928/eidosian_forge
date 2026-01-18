from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
class SingleTermManager(TermManagerBase):
    """All connections to the websocket share a common terminal."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the manager."""
        super().__init__(**kwargs)
        self.terminal: PtyWithClients | None = None

    def get_terminal(self, url_component: Any=None) -> PtyWithClients:
        """ "Get the singleton terminal."""
        if self.terminal is None:
            self.terminal = self.new_terminal()
            self.start_reading(self.terminal)
        return self.terminal

    async def kill_all(self) -> None:
        """Kill the singletone terminal."""
        await super().kill_all()
        self.terminal = None