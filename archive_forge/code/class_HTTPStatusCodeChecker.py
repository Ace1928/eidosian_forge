import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class HTTPStatusCodeChecker(BaseChecker):

    def __init__(self, status_code):
        self._status_code = status_code

    def _check_response(self, attempt_number, response):
        if response[0].status_code == self._status_code:
            logger.debug('retry needed: retryable HTTP status code received: %s', self._status_code)
            return True
        else:
            return False