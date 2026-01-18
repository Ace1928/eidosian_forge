import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def _create_single_response_checker(response):
    if 'service_error_code' in response:
        checker = ServiceErrorCodeChecker(status_code=response['http_status_code'], error_code=response['service_error_code'])
    elif 'http_status_code' in response:
        checker = HTTPStatusCodeChecker(status_code=response['http_status_code'])
    elif 'crc32body' in response:
        checker = CRC32Checker(header=response['crc32body'])
    else:
        raise ValueError('Unknown retry policy')
    return checker