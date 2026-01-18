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
class UniqueTermManager(TermManagerBase):
    """Give each websocket a unique terminal to use."""

    def __init__(self, max_terminals: int | None=None, **kwargs: Any) -> None:
        """Initialize the manager."""
        super().__init__(**kwargs)
        self.max_terminals = max_terminals

    def get_terminal(self, url_component: Any=None) -> PtyWithClients:
        """Get a terminal from the manager."""
        if self.max_terminals and len(self.ptys_by_fd) >= self.max_terminals:
            raise MaxTerminalsReached(self.max_terminals)
        term = self.new_terminal()
        self.start_reading(term)
        return term

    def client_disconnected(self, websocket: TermSocket) -> None:
        """Send terminal SIGHUP when client disconnects."""
        self.log.info('Websocket closed, sending SIGHUP to terminal.')
        if websocket.terminal:
            if os.name == 'nt':
                websocket.terminal.kill()
                self.pty_read(websocket.terminal.ptyproc.fd)
                return
            websocket.terminal.killpg(signal.SIGHUP)