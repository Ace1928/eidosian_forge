from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _set_table_ttl(self):
    """Enable or disable Time to Live on the table."""
    description = self._get_table_ttl_description()
    status = description['TimeToLiveDescription']['TimeToLiveStatus']
    if status in ('ENABLED', 'ENABLING'):
        cur_attr_name = description['TimeToLiveDescription']['AttributeName']
        if self._has_ttl():
            if cur_attr_name == self._ttl_field.name:
                logger.debug('DynamoDB Time to Live is {situation} on table {table}'.format(situation='already enabled' if status == 'ENABLED' else 'currently being enabled', table=self.table_name))
                return description
    elif status in ('DISABLED', 'DISABLING'):
        if not self._has_ttl():
            logger.debug('DynamoDB Time to Live is {situation} on table {table}'.format(situation='already disabled' if status == 'DISABLED' else 'currently being disabled', table=self.table_name))
            return description
    else:
        logger.warning('Unknown DynamoDB Time to Live status {status} on table {table}. Attempting to continue.'.format(status=status, table=self.table_name))
    attr_name = cur_attr_name if status == 'ENABLED' else self._ttl_field.name
    try:
        specification = self._client.update_time_to_live(**self._get_ttl_specification(ttl_attr_name=attr_name))
        logger.info('DynamoDB table Time to Live updated: table={table} enabled={enabled} attribute={attr}'.format(table=self.table_name, enabled=self._has_ttl(), attr=self._ttl_field.name))
        return specification
    except ClientError as e:
        error_code = e.response['Error'].get('Code', 'Unknown')
        error_message = e.response['Error'].get('Message', 'Unknown')
        logger.error('Error {action} Time to Live on DynamoDB table {table}: {code}: {message}'.format(action='enabling' if self._has_ttl() else 'disabling', table=self.table_name, code=error_code, message=error_message))
        raise e