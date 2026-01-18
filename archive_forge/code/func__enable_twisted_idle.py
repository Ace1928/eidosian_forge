from __future__ import annotations
import functools
import logging
import sys
import typing
from twisted.internet.abstract import FileDescriptor
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from .abstract_loop import EventLoop, ExitMainLoop
def _enable_twisted_idle(self) -> None:
    """
        Twisted's reactors don't have an idle or enter-idle callback
        so the best we can do for now is to set a timer event in a very
        short time to approximate an enter-idle callback.

        .. WARNING::
           This will perform worse than the other event loops until we can find a
           fix or workaround
        """
    if self._twisted_idle_enabled:
        return
    self.reactor.callLater(self._idle_emulation_delay, self.handle_exit(self._twisted_idle_callback, enable_idle=False))
    self._twisted_idle_enabled = True