from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _get_ttl_specification(self, ttl_attr_name):
    """Get the boto3 structure describing the DynamoDB TTL specification."""
    return {'TableName': self.table_name, 'TimeToLiveSpecification': {'Enabled': self._has_ttl(), 'AttributeName': ttl_attr_name}}