import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def _retry_handler(self, response, i, next_sleep):
    status = None
    if response.status == 400:
        response_body = response.read().decode('utf-8')
        boto.log.debug(response_body)
        data = json.loads(response_body)
        if self.ThruputError in data.get('__type'):
            self.throughput_exceeded_events += 1
            msg = '%s, retry attempt %s' % (self.ThruputError, i)
            next_sleep = self._exponential_time(i)
            i += 1
            status = (msg, i, next_sleep)
            if i == self.NumberRetries:
                raise dynamodb_exceptions.DynamoDBThroughputExceededError(response.status, response.reason, data)
        elif self.SessionExpiredError in data.get('__type'):
            msg = 'Renewing Session Token'
            self._get_session_token()
            status = (msg, i + self.num_retries - 1, 0)
        elif self.ConditionalCheckFailedError in data.get('__type'):
            raise dynamodb_exceptions.DynamoDBConditionalCheckFailedError(response.status, response.reason, data)
        elif self.ValidationError in data.get('__type'):
            raise dynamodb_exceptions.DynamoDBValidationError(response.status, response.reason, data)
        else:
            raise self.ResponseError(response.status, response.reason, data)
    expected_crc32 = response.getheader('x-amz-crc32')
    if self._validate_checksums and expected_crc32 is not None:
        boto.log.debug('Validating crc32 checksum for body: %s', response.read().decode('utf-8'))
        actual_crc32 = crc32(response.read()) & 4294967295
        expected_crc32 = int(expected_crc32)
        if actual_crc32 != expected_crc32:
            msg = 'The calculated checksum %s did not match the expected checksum %s' % (actual_crc32, expected_crc32)
            status = (msg, i + 1, self._exponential_time(i))
    return status