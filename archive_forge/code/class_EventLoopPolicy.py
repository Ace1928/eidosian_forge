import asyncio as __asyncio
import typing as _typing
import sys as _sys
import warnings as _warnings
from asyncio.events import BaseDefaultEventLoopPolicy as __BasePolicy
from . import includes as __includes  # NOQA
from .loop import Loop as __BaseLoop  # NOQA
from ._version import __version__  # NOQA
class EventLoopPolicy(__BasePolicy):
    """Event loop policy.

    The preferred way to make your application use uvloop:

    >>> import asyncio
    >>> import uvloop
    >>> asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    >>> asyncio.get_event_loop()
    <uvloop.Loop running=False closed=False debug=False>
    """

    def _loop_factory(self) -> Loop:
        return new_event_loop()
    if _typing.TYPE_CHECKING:

        def get_child_watcher(self) -> _typing.NoReturn:
            ...

        def set_child_watcher(self, watcher: _typing.Any) -> _typing.NoReturn:
            ...