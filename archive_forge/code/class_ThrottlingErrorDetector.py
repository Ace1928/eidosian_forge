import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class ThrottlingErrorDetector:

    def __init__(self, retry_event_adapter):
        self._modeled_error_detector = ModeledRetryErrorDetector()
        self._fixed_error_code_detector = ThrottledRetryableChecker()
        self._retry_event_adapter = retry_event_adapter

    def is_throttling_error(self, **kwargs):
        context = self._retry_event_adapter.create_retry_context(**kwargs)
        if self._fixed_error_code_detector.is_retryable(context):
            return True
        error_type = self._modeled_error_detector.detect_error_type(context)
        return error_type == self._modeled_error_detector.THROTTLING_ERROR