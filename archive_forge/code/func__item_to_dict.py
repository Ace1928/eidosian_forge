from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _item_to_dict(self, raw_response):
    """Convert get_item() response to field-value pairs."""
    if 'Item' not in raw_response:
        return {}
    return {field.name: raw_response['Item'][field.name][field.data_type] for field in self._available_fields}