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
def _origin_verified(self, request):
    request_origin = request.META['HTTP_ORIGIN']
    try:
        good_host = request.get_host()
    except DisallowedHost:
        pass
    else:
        good_origin = '%s://%s' % ('https' if request.is_secure() else 'http', good_host)
        if request_origin == good_origin:
            return True
    if request_origin in self.allowed_origins_exact:
        return True
    try:
        parsed_origin = urlparse(request_origin)
    except ValueError:
        return False
    request_scheme = parsed_origin.scheme
    request_netloc = parsed_origin.netloc
    return any((is_same_domain(request_netloc, host) for host in self.allowed_origin_subdomains.get(request_scheme, ())))