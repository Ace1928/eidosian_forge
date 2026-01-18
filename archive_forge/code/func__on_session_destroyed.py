from __future__ import annotations
import logging # isort:skip
from ..application import SessionContext
from .lifecycle import LifecycleHandler
def _on_session_destroyed(session_context: SessionContext) -> None:
    """
    Calls any on_session_destroyed callbacks defined on the Document
    """
    callbacks = session_context._document.session_destroyed_callbacks
    session_context._document.session_destroyed_callbacks = set()
    for callback in callbacks:
        try:
            callback(session_context)
        except Exception as e:
            log.warning(f'DocumentLifeCycleHandler on_session_destroyed callback {callback} failed with following error: {e}')
    if callbacks:
        del callback
        del callbacks
        import gc
        gc.collect()