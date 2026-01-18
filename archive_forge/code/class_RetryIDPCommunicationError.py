import logging
from binascii import crc32
from botocore.retries.base import BaseRetryableChecker
class RetryIDPCommunicationError(BaseRetryableChecker):
    _SERVICE_NAME = 'sts'

    def is_retryable(self, context):
        service_name = context.operation_model.service_model.service_name
        if service_name != self._SERVICE_NAME:
            return False
        error_code = context.get_error_code()
        return error_code == 'IDPCommunicationError'