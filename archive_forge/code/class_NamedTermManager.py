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
class NamedTermManager(TermManagerBase):
    """Share terminals between websockets connected to the same endpoint."""

    def __init__(self, max_terminals: Any=None, **kwargs: Any) -> None:
        """Initialize the manager."""
        super().__init__(**kwargs)
        self.max_terminals = max_terminals
        self.terminals: dict[str, PtyWithClients] = {}

    def get_terminal(self, term_name: str) -> PtyWithClients:
        """Get or create a terminal by name."""
        assert term_name is not None
        if term_name in self.terminals:
            return self.terminals[term_name]
        if self.max_terminals and len(self.terminals) >= self.max_terminals:
            raise MaxTerminalsReached(self.max_terminals)
        self.log.info('New terminal with specified name: %s', term_name)
        term = self.new_terminal()
        term.term_name = term_name
        self.terminals[term_name] = term
        self.start_reading(term)
        return term
    name_template = '%d'

    def _next_available_name(self) -> str | None:
        for n in itertools.count(start=1):
            name = self.name_template % n
            if name not in self.terminals:
                return name
        return None

    def new_named_terminal(self, **kwargs: Any) -> tuple[str, PtyWithClients]:
        """Create a new named terminal with an automatic name."""
        name = kwargs['name'] if 'name' in kwargs else self._next_available_name()
        term = self.new_terminal(**kwargs)
        self.log.info('New terminal with automatic name: %s', name)
        term.term_name = name
        self.terminals[name] = term
        self.start_reading(term)
        return (name, term)

    def kill(self, name: str, sig: int=signal.SIGTERM) -> None:
        """Kill a terminal by name."""
        term = self.terminals[name]
        term.kill(sig)

    async def terminate(self, name: str, force: bool=False) -> None:
        """Terminate a terminal by name."""
        term = self.terminals[name]
        await term.terminate(force=force)

    def on_eof(self, ptywclients: PtyWithClients) -> None:
        """Handle end of file for a pty with clients."""
        super().on_eof(ptywclients)
        name = ptywclients.term_name
        self.log.info('Terminal %s closed', name)
        assert name is not None
        self.terminals.pop(name, None)

    async def kill_all(self) -> None:
        """Kill all terminals."""
        await super().kill_all()
        self.terminals = {}