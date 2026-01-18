from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
def _cancel_scope(self, scope: trio.CancelScope) -> bool:
    """Cancels the given Trio cancellation scope.

        Returns:
            True if the scope was cancelled, False if it was cancelled already
            before invoking this function
        """
    existed = not scope.cancel_called
    scope.cancel()
    return existed