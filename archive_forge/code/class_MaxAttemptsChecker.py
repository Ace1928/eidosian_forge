import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class MaxAttemptsChecker(BaseRetryableChecker):

    def __init__(self, max_attempts):
        self._max_attempts = max_attempts

    def is_retryable(self, context):
        under_max_attempts = context.attempt_number < self._max_attempts
        retries_context = context.request_context.get('retries')
        if retries_context:
            retries_context['max'] = max(retries_context.get('max', 0), self._max_attempts)
        if not under_max_attempts:
            logger.debug('Max attempts of %s reached.', self._max_attempts)
            context.add_retry_metadata(MaxAttemptsReached=True)
        return under_max_attempts