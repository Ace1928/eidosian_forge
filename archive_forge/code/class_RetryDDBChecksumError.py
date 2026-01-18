import logging
from binascii import crc32
from botocore.retries.base import BaseRetryableChecker
class RetryDDBChecksumError(BaseRetryableChecker):
    _CHECKSUM_HEADER = 'x-amz-crc32'
    _SERVICE_NAME = 'dynamodb'

    def is_retryable(self, context):
        service_name = context.operation_model.service_model.service_name
        if service_name != self._SERVICE_NAME:
            return False
        if context.http_response is None:
            return False
        checksum = context.http_response.headers.get(self._CHECKSUM_HEADER)
        if checksum is None:
            return False
        actual_crc32 = crc32(context.http_response.content) & 4294967295
        if actual_crc32 != int(checksum):
            logger.debug('DynamoDB crc32 checksum does not match, expected: %s, actual: %s', checksum, actual_crc32)
            return True