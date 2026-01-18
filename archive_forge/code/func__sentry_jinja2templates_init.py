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
def _sentry_jinja2templates_init(self, *args, **kwargs):

    def add_sentry_trace_meta(request):
        hub = Hub.current
        trace_meta = Markup(hub.trace_propagation_meta())
        return {'sentry_trace_meta': trace_meta}
    kwargs.setdefault('context_processors', [])
    if add_sentry_trace_meta not in kwargs['context_processors']:
        kwargs['context_processors'].append(add_sentry_trace_meta)
    return old_jinja2templates_init(self, *args, **kwargs)