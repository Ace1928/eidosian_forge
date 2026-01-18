from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
def _handle_main_loop_exception(self, exc: BaseException) -> None:
    """Handles exceptions raised from the main loop, catching ExitMainLoop
        instead of letting it propagate through.

        Note that since Trio may collect multiple exceptions from tasks into a
        Trio MultiError, we cannot simply use a try..catch clause, we need a
        helper function like this.
        """
    self._idle_callbacks.clear()
    if isinstance(exc, exceptiongroup.BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    if isinstance(exc, ExitMainLoop):
        return
    raise exc.with_traceback(exc.__traceback__) from None