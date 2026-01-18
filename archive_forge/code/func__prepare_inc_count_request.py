from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _prepare_inc_count_request(self, key: str) -> Dict[str, Any]:
    """Construct the counter increment request parameters"""
    return {'TableName': self.table_name, 'Key': {self._key_field.name: {self._key_field.data_type: key}}, 'UpdateExpression': f'set {self._count_filed.name} = {self._count_filed.name} + :num', 'ExpressionAttributeValues': {':num': {'N': '1'}}, 'ReturnValues': 'UPDATED_NEW'}