import datetime
import io
import json
import mimetypes
import os
import re
import sys
import time
import warnings
from email.header import Header
from http.client import responses
from urllib.parse import urlparse
from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.core import signals, signing
from django.core.exceptions import DisallowedRedirect
from django.core.serializers.json import DjangoJSONEncoder
from django.http.cookie import SimpleCookie
from django.utils import timezone
from django.utils.datastructures import CaseInsensitiveMapping
from django.utils.encoding import iri_to_uri
from django.utils.http import content_disposition_header, http_date
from django.utils.regex_helper import _lazy_re_compile
def _convert_to_charset(self, value, charset, mime_encode=False):
    """
        Convert headers key/value to ascii/latin-1 native strings.
        `charset` must be 'ascii' or 'latin-1'. If `mime_encode` is True and
        `value` can't be represented in the given charset, apply MIME-encoding.
        """
    try:
        if isinstance(value, str):
            value.encode(charset)
        elif isinstance(value, bytes):
            value = value.decode(charset)
        else:
            value = str(value)
            value.encode(charset)
        if '\n' in value or '\r' in value:
            raise BadHeaderError(f"Header values can't contain newlines (got {value!r})")
    except UnicodeError as e:
        if isinstance(value, bytes) and (b'\n' in value or b'\r' in value) or (isinstance(value, str) and ('\n' in value or '\r' in value)):
            raise BadHeaderError(f"Header values can't contain newlines (got {value!r})") from e
        if mime_encode:
            value = Header(value, 'utf-8', maxlinelen=sys.maxsize).encode()
        else:
            e.reason += ', HTTP response headers must be in %s format' % charset
            raise
    return value