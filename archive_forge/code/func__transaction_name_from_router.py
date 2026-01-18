from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def _transaction_name_from_router(scope):
    router = scope.get('router')
    if not router:
        return None
    for route in router.routes:
        match = route.matches(scope)
        if match[0] == Match.FULL:
            return route.path
    return None