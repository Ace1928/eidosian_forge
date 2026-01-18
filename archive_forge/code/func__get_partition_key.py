from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
@classmethod
def _get_partition_key(cls, key):
    if not key or key.isspace():
        raise ValueError('Key cannot be none, empty or whitespace.')
    return {'partitionKey': key}