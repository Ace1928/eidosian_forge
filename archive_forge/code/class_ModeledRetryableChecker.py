import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class ModeledRetryableChecker(BaseRetryableChecker):
    """Check if an error has been modeled as retryable."""

    def __init__(self):
        self._error_detector = ModeledRetryErrorDetector()

    def is_retryable(self, context):
        error_code = context.get_error_code()
        if error_code is None:
            return False
        return self._error_detector.detect_error_type(context) is not None