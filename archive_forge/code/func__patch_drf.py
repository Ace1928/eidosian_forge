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
def _patch_drf():
    """
    Patch Django Rest Framework for more/better request data. DRF's request
    type is a wrapper around Django's request type. The attribute we're
    interested in is `request.data`, which is a cached property containing a
    parsed request body. Reading a request body from that property is more
    reliable than reading from any of Django's own properties, as those don't
    hold payloads in memory and therefore can only be accessed once.

    We patch the Django request object to include a weak backreference to the
    DRF request object, such that we can later use either in
    `DjangoRequestExtractor`.

    This function is not called directly on SDK setup, because importing almost
    any part of Django Rest Framework will try to access Django settings (where
    `sentry_sdk.init()` might be called from in the first place). Instead we
    run this function on every request and do the patching on the first
    request.
    """
    global _DRF_PATCHED
    if _DRF_PATCHED:
        return
    with _DRF_PATCH_LOCK:
        if _DRF_PATCHED:
            return
        _DRF_PATCHED = True
        with capture_internal_exceptions():
            try:
                from rest_framework.views import APIView
            except ImportError:
                pass
            else:
                old_drf_initial = APIView.initial

                def sentry_patched_drf_initial(self, request, *args, **kwargs):
                    with capture_internal_exceptions():
                        request._request._sentry_drf_request_backref = weakref.ref(request)
                        pass
                    return old_drf_initial(self, request, *args, **kwargs)
                APIView.initial = sentry_patched_drf_initial