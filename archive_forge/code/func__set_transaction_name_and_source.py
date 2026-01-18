from __future__ import absolute_import
import inspect
import sys
import threading
import weakref
from importlib import import_module
from sentry_sdk._compat import string_types, text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.db.explain_plan.django import attach_explain_plan_to_span
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.serializer import add_global_repr_processor
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_URL
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.django.transactions import LEGACY_RESOLVER
from sentry_sdk.integrations.django.templates import (
from sentry_sdk.integrations.django.middleware import patch_django_middlewares
from sentry_sdk.integrations.django.signals_handlers import patch_signals
from sentry_sdk.integrations.django.views import patch_views
def _set_transaction_name_and_source(scope, transaction_style, request):
    try:
        transaction_name = None
        if transaction_style == 'function_name':
            fn = resolve(request.path).func
            transaction_name = transaction_from_function(getattr(fn, 'view_class', fn))
        elif transaction_style == 'url':
            if hasattr(request, 'urlconf'):
                transaction_name = LEGACY_RESOLVER.resolve(request.path_info, urlconf=request.urlconf)
            else:
                transaction_name = LEGACY_RESOLVER.resolve(request.path_info)
        if transaction_name is None:
            transaction_name = request.path_info
            source = TRANSACTION_SOURCE_URL
        else:
            source = SOURCE_FOR_STYLE[transaction_style]
        scope.set_transaction_name(transaction_name, source=source)
    except Resolver404:
        urlconf = import_module(settings.ROOT_URLCONF)
        if hasattr(urlconf, 'handler404'):
            handler = urlconf.handler404
            if isinstance(handler, string_types):
                scope.transaction = handler
            else:
                scope.transaction = transaction_from_function(getattr(handler, 'view_class', handler))
    except Exception:
        pass