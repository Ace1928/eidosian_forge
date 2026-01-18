import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def _exponential_time(self, i):
    if i == 0:
        next_sleep = 0
    else:
        next_sleep = min(0.05 * 2 ** i, boto.config.get('Boto', 'max_retry_delay', 60))
    return next_sleep