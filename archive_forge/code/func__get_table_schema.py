from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _get_table_schema(self):
    """Get the boto3 structure describing the DynamoDB table schema."""
    return {'AttributeDefinitions': [{'AttributeName': self._key_field.name, 'AttributeType': self._key_field.data_type}], 'TableName': self.table_name, 'KeySchema': [{'AttributeName': self._key_field.name, 'KeyType': 'HASH'}], 'ProvisionedThroughput': {'ReadCapacityUnits': self.read_capacity_units, 'WriteCapacityUnits': self.write_capacity_units}}