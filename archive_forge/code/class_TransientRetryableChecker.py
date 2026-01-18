import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class TransientRetryableChecker(BaseRetryableChecker):
    _TRANSIENT_ERROR_CODES = ['RequestTimeout', 'RequestTimeoutException', 'PriorRequestNotComplete']
    _TRANSIENT_STATUS_CODES = [500, 502, 503, 504]
    _TRANSIENT_EXCEPTION_CLS = (ConnectionError, HTTPClientError)

    def __init__(self, transient_error_codes=None, transient_status_codes=None, transient_exception_cls=None):
        if transient_error_codes is None:
            transient_error_codes = self._TRANSIENT_ERROR_CODES[:]
        if transient_status_codes is None:
            transient_status_codes = self._TRANSIENT_STATUS_CODES[:]
        if transient_exception_cls is None:
            transient_exception_cls = self._TRANSIENT_EXCEPTION_CLS
        self._transient_error_codes = transient_error_codes
        self._transient_status_codes = transient_status_codes
        self._transient_exception_cls = transient_exception_cls

    def is_retryable(self, context):
        if context.get_error_code() in self._transient_error_codes:
            return True
        if context.http_response is not None:
            if context.http_response.status_code in self._transient_status_codes:
                return True
        if context.caught_exception is not None:
            return isinstance(context.caught_exception, self._transient_exception_cls)
        return False