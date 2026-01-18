import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class RetryQuotaChecker:
    _RETRY_COST = 5
    _NO_RETRY_INCREMENT = 1
    _TIMEOUT_RETRY_REQUEST = 10
    _TIMEOUT_EXCEPTIONS = (ConnectTimeoutError, ReadTimeoutError)

    def __init__(self, quota):
        self._quota = quota
        self._last_amount_acquired = None

    def acquire_retry_quota(self, context):
        if self._is_timeout_error(context):
            capacity_amount = self._TIMEOUT_RETRY_REQUEST
        else:
            capacity_amount = self._RETRY_COST
        success = self._quota.acquire(capacity_amount)
        if success:
            context.request_context['retry_quota_capacity'] = capacity_amount
            return True
        context.add_retry_metadata(RetryQuotaReached=True)
        return False

    def _is_timeout_error(self, context):
        return isinstance(context.caught_exception, self._TIMEOUT_EXCEPTIONS)

    def release_retry_quota(self, context, http_response, **kwargs):
        if http_response is None:
            return
        status_code = http_response.status_code
        if 200 <= status_code < 300:
            if 'retry_quota_capacity' not in context:
                self._quota.release(self._NO_RETRY_INCREMENT)
            else:
                capacity_amount = context['retry_quota_capacity']
                self._quota.release(capacity_amount)