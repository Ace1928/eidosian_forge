import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
def _patch_cache(cache):
    if not hasattr(cache, '_sentry_patched'):
        for method_name in METHODS_TO_INSTRUMENT:
            _patch_cache_method(cache, method_name)
        cache._sentry_patched = True