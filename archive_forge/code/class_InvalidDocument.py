from datetime import datetime, timedelta, timezone
from kombu.exceptions import EncodeError
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url, urlparse
from celery import states
from celery.exceptions import ImproperlyConfigured
from .base import BaseBackend
class InvalidDocument(Exception):
    pass