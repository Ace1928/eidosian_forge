from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import parse_url
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
def _init_from_params(self, hostname, port, virtual_host, **params):
    logger.debug('Setting on Consul client to connect to %s:%d', hostname, port)
    self.path = virtual_host
    self.hostname = hostname
    self.port = port
    if params.get('one_client', None):
        self.one_client = self.client()