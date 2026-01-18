from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _prepare_init_count_request(self, key: str) -> Dict[str, Any]:
    """Construct the counter initialization request parameters"""
    timestamp = time()
    return {'TableName': self.table_name, 'Item': {self._key_field.name: {self._key_field.data_type: key}, self._count_filed.name: {self._count_filed.data_type: '0'}, self._timestamp_field.name: {self._timestamp_field.data_type: str(timestamp)}}}