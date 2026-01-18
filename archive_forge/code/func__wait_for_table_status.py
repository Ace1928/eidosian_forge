from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _wait_for_table_status(self, expected='ACTIVE'):
    """Poll for the expected table status."""
    achieved_state = False
    while not achieved_state:
        table_description = self.client.describe_table(TableName=self.table_name)
        logger.debug('Waiting for DynamoDB table {} to become {}.'.format(self.table_name, expected))
        current_status = table_description['Table']['TableStatus']
        achieved_state = current_status == expected
        sleep(1)