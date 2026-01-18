from datetime import datetime, timedelta, timezone
from kombu.exceptions import EncodeError
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url, urlparse
from celery import states
from celery.exceptions import ImproperlyConfigured
from .base import BaseBackend
@staticmethod
def _ensure_mongodb_uri_compliance(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith('mongodb'):
        url = f'mongodb+{url}'
    if url == 'mongodb://':
        url += 'localhost'
    return url