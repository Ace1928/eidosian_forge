import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class ServiceErrorCodeChecker(BaseChecker):

    def __init__(self, status_code, error_code):
        self._status_code = status_code
        self._error_code = error_code

    def _check_response(self, attempt_number, response):
        if response[0].status_code == self._status_code:
            actual_error_code = response[1].get('Error', {}).get('Code')
            if actual_error_code == self._error_code:
                logger.debug('retry needed: matching HTTP status and error code seen: %s, %s', self._status_code, self._error_code)
                return True
        return False