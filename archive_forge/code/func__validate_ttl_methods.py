from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _validate_ttl_methods(self):
    """Verify boto support for the DynamoDB Time to Live methods."""
    required_methods = ('update_time_to_live', 'describe_time_to_live')
    missing_methods = []
    for method in list(required_methods):
        if not hasattr(self._client, method):
            missing_methods.append(method)
    if missing_methods:
        logger.error('boto3 method(s) {methods} not found; ensure that boto3>=1.9.178 and botocore>=1.12.178 are installed'.format(methods=','.join(missing_methods)))
        raise AttributeError('boto3 method(s) {methods} not found'.format(methods=','.join(missing_methods)))