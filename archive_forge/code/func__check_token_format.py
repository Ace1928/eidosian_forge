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
def _check_token_format(token):
    """
    Raise an InvalidTokenFormat error if the token has an invalid length or
    characters that aren't allowed. The token argument can be a CSRF cookie
    secret or non-cookie CSRF token, and either masked or unmasked.
    """
    if len(token) not in (CSRF_TOKEN_LENGTH, CSRF_SECRET_LENGTH):
        raise InvalidTokenFormat(REASON_INCORRECT_LENGTH)
    if invalid_token_chars_re.search(token):
        raise InvalidTokenFormat(REASON_INVALID_CHARACTERS)