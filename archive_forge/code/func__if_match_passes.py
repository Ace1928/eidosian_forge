import time
from collections import defaultdict
from hashlib import md5
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.http import http_date, parse_etags, parse_http_date_safe, quote_etag
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language
def _if_match_passes(target_etag, etags):
    """
    Test the If-Match comparison as defined in RFC 9110 Section 13.1.1.
    """
    if not target_etag:
        return False
    elif etags == ['*']:
        return True
    elif target_etag.startswith('W/'):
        return False
    else:
        return target_etag in etags