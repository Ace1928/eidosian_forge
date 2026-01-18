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
def _patch_get_response():
    """
    patch get_response, because at that point we have the Django request object
    """
    from django.core.handlers.base import BaseHandler
    old_get_response = BaseHandler.get_response

    def sentry_patched_get_response(self, request):
        _before_get_response(request)
        rv = old_get_response(self, request)
        _after_get_response(request)
        return rv
    BaseHandler.get_response = sentry_patched_get_response
    if hasattr(BaseHandler, 'get_response_async'):
        from sentry_sdk.integrations.django.asgi import patch_get_response_async
        patch_get_response_async(BaseHandler, _before_get_response)