from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import parse_url
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
def _key_to_consul_key(self, key):
    key = bytes_to_str(key)
    return key if self.path is None else f'{self.path}/{key}'