from __future__ import absolute_import
import sys
import types
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub
from sentry_sdk._compat import reraise
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def _wrap_task_call(func):
    """
    Wrap task call with a try catch to get exceptions.
    Pass the client on to raise_exception so it can get rebinded.
    """
    client = Hub.current.client

    @wraps(func)
    def _inner(*args, **kwargs):
        try:
            gen = func(*args, **kwargs)
        except Exception:
            raise_exception(client)
        if not isinstance(gen, types.GeneratorType):
            return gen
        return _wrap_generator_call(gen, client)
    setattr(_inner, USED_FUNC, True)
    return _inner