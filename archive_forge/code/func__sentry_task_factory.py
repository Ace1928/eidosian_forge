from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import event_from_exception
def _sentry_task_factory(loop, coro, **kwargs):

    async def _coro_creating_hub_and_span():
        hub = Hub(Hub.current)
        result = None
        with hub:
            with hub.start_span(op=OP.FUNCTION, description=get_name(coro)):
                try:
                    result = await coro
                except Exception:
                    reraise(*_capture_exception(hub))
        return result
    if orig_task_factory:
        return orig_task_factory(loop, _coro_creating_hub_and_span(), **kwargs)
    task = Task(_coro_creating_hub_and_span(), loop=loop, **kwargs)
    if task._source_traceback:
        del task._source_traceback[-1]
    return task