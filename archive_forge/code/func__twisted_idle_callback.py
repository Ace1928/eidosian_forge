from __future__ import annotations
import functools
import logging
import sys
import typing
from twisted.internet.abstract import FileDescriptor
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from .abstract_loop import EventLoop, ExitMainLoop
def _twisted_idle_callback(self) -> None:
    for callback in self._idle_callbacks.values():
        callback()
    self._twisted_idle_enabled = False