import logging
import string
from collections import defaultdict
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.http import HttpHeaders, UnreadablePostError
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
def _set_csrf_cookie(self, request, response):
    if settings.CSRF_USE_SESSIONS:
        if request.session.get(CSRF_SESSION_KEY) != request.META['CSRF_COOKIE']:
            request.session[CSRF_SESSION_KEY] = request.META['CSRF_COOKIE']
    else:
        response.set_cookie(settings.CSRF_COOKIE_NAME, request.META['CSRF_COOKIE'], max_age=settings.CSRF_COOKIE_AGE, domain=settings.CSRF_COOKIE_DOMAIN, path=settings.CSRF_COOKIE_PATH, secure=settings.CSRF_COOKIE_SECURE, httponly=settings.CSRF_COOKIE_HTTPONLY, samesite=settings.CSRF_COOKIE_SAMESITE)
        patch_vary_headers(response, ('Cookie',))