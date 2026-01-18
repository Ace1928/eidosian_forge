from functools import wraps
from .cloud import CloudRetry
class AWSRetry(CloudRetry):
    base_class = _botocore_exception_maybe()

    @staticmethod
    def status_code_from_exception(error):
        return error.response['Error']['Code']

    @staticmethod
    def found(response_code, catch_extra_error_codes=None):
        retry_on = ['RequestLimitExceeded', 'Unavailable', 'ServiceUnavailable', 'InternalFailure', 'InternalError', 'TooManyRequestsException', 'Throttling']
        if catch_extra_error_codes:
            retry_on.extend(catch_extra_error_codes)
        return response_code in retry_on