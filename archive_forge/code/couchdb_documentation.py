from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
Connect to the CouchDB server.