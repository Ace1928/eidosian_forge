from __future__ import annotations
import asyncio
import os
import select
import selectors
import sys
import threading
from asyncio import AbstractEventLoop, get_running_loop
from selectors import BaseSelector, SelectorKey
from typing import TYPE_CHECKING, Any, Callable, Mapping
class InputHookSelector(BaseSelector):
    """
    Usage:

        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(InputHookSelector(selector, inputhook))
        asyncio.set_event_loop(loop)
    """

    def __init__(self, selector: BaseSelector, inputhook: Callable[[InputHookContext], None]) -> None:
        self.selector = selector
        self.inputhook = inputhook
        self._r, self._w = os.pipe()

    def register(self, fileobj: FileDescriptorLike, events: _EventMask, data: Any=None) -> SelectorKey:
        return self.selector.register(fileobj, events, data=data)

    def unregister(self, fileobj: FileDescriptorLike) -> SelectorKey:
        return self.selector.unregister(fileobj)

    def modify(self, fileobj: FileDescriptorLike, events: _EventMask, data: Any=None) -> SelectorKey:
        return self.selector.modify(fileobj, events, data=None)

    def select(self, timeout: float | None=None) -> list[tuple[SelectorKey, _EventMask]]:
        if len(getattr(get_running_loop(), '_ready', [])) > 0:
            return self.selector.select(timeout=timeout)
        ready = False
        result = None

        def run_selector() -> None:
            nonlocal ready, result
            result = self.selector.select(timeout=timeout)
            os.write(self._w, b'x')
            ready = True
        th = threading.Thread(target=run_selector)
        th.start()

        def input_is_ready() -> bool:
            return ready
        self.inputhook(InputHookContext(self._r, input_is_ready))
        try:
            if sys.platform != 'win32':
                select.select([self._r], [], [], None)
            os.read(self._r, 1024)
        except OSError:
            pass
        th.join()
        assert result is not None
        return result

    def close(self) -> None:
        """
        Clean up resources.
        """
        if self._r:
            os.close(self._r)
            os.close(self._w)
        self._r = self._w = -1
        self.selector.close()

    def get_map(self) -> Mapping[FileDescriptorLike, SelectorKey]:
        return self.selector.get_map()