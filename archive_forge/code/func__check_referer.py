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
def _check_referer(self, request):
    referer = request.META.get('HTTP_REFERER')
    if referer is None:
        raise RejectRequest(REASON_NO_REFERER)
    try:
        referer = urlparse(referer)
    except ValueError:
        raise RejectRequest(REASON_MALFORMED_REFERER)
    if '' in (referer.scheme, referer.netloc):
        raise RejectRequest(REASON_MALFORMED_REFERER)
    if referer.scheme != 'https':
        raise RejectRequest(REASON_INSECURE_REFERER)
    if any((is_same_domain(referer.netloc, host) for host in self.csrf_trusted_origins_hosts)):
        return
    good_referer = settings.SESSION_COOKIE_DOMAIN if settings.CSRF_USE_SESSIONS else settings.CSRF_COOKIE_DOMAIN
    if good_referer is None:
        try:
            good_referer = request.get_host()
        except DisallowedHost:
            raise RejectRequest(REASON_BAD_REFERER % referer.geturl())
    else:
        server_port = request.get_port()
        if server_port not in ('443', '80'):
            good_referer = '%s:%s' % (good_referer, server_port)
    if not is_same_domain(referer.netloc, good_referer):
        raise RejectRequest(REASON_BAD_REFERER % referer.geturl())