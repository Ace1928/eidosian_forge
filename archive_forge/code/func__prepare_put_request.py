from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _prepare_put_request(self, key, value):
    """Construct the item creation request parameters."""
    timestamp = time()
    put_request = {'TableName': self.table_name, 'Item': {self._key_field.name: {self._key_field.data_type: key}, self._value_field.name: {self._value_field.data_type: value}, self._timestamp_field.name: {self._timestamp_field.data_type: str(timestamp)}}}
    if self._has_ttl():
        put_request['Item'].update({self._ttl_field.name: {self._ttl_field.data_type: str(int(timestamp + self.time_to_live_seconds))}})
    return put_request