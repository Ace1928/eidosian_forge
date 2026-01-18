from binascii import crc32
import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.dynamodb2 import exceptions
def _truncated_exponential_time(self, i):
    if i == 0:
        next_sleep = 0
    else:
        next_sleep = min(0.05 * 2 ** i, boto.config.get('Boto', 'max_retry_delay', 60))
    return next_sleep