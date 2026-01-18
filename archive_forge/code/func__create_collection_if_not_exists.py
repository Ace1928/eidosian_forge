from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _create_collection_if_not_exists(self, client):
    try:
        client.CreateCollection(self._database_link, {'id': self._collection_name, 'partitionKey': {'paths': ['/id'], 'kind': PartitionKind.Hash}})
    except HTTPFailure as ex:
        if ex.status_code != ERROR_EXISTS:
            raise
    else:
        LOGGER.info('Created CosmosDB collection %s/%s', self._database_name, self._collection_name)