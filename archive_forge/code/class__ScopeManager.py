import copy
import sys
from contextlib import contextmanager
from sentry_sdk._compat import with_metaclass
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import Scope
from sentry_sdk.client import Client
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
class _ScopeManager(object):

    def __init__(self, hub):
        self._hub = hub
        self._original_len = len(hub._stack)
        self._layer = hub._stack[-1]

    def __enter__(self):
        scope = self._layer[1]
        assert scope is not None
        return scope

    def __exit__(self, exc_type, exc_value, tb):
        current_len = len(self._hub._stack)
        if current_len < self._original_len:
            logger.error('Scope popped too soon. Popped %s scopes too many.', self._original_len - current_len)
            return
        elif current_len > self._original_len:
            logger.warning('Leaked %s scopes: %s', current_len - self._original_len, self._hub._stack[self._original_len:])
        layer = self._hub._stack[self._original_len - 1]
        del self._hub._stack[self._original_len - 1:]
        if layer[1] != self._layer[1]:
            logger.error('Wrong scope found. Meant to pop %s, but popped %s.', layer[1], self._layer[1])
        elif layer[0] != self._layer[0]:
            warning = 'init() called inside of pushed scope. This might be entirely legitimate but usually occurs when initializing the SDK inside a request handler or task/job function. Try to initialize the SDK as early as possible instead.'
            logger.warning(warning)