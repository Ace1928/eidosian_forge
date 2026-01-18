import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class RetryContext:
    """Normalize a response that we use to check if a retry should occur.

    This class smoothes over the different types of responses we may get
    from a service including:

        * A modeled error response from the service that contains a service
          code and error message.
        * A raw HTTP response that doesn't contain service protocol specific
          error keys.
        * An exception received while attempting to retrieve a response.
          This could be a ConnectionError we receive from our HTTP layer which
          could represent that we weren't able to receive a response from
          the service.

    This class guarantees that at least one of the above attributes will be
    non None.

    This class is meant to provide a read-only view into the properties
    associated with a possible retryable response.  None of the properties
    are meant to be modified directly.

    """

    def __init__(self, attempt_number, operation_model=None, parsed_response=None, http_response=None, caught_exception=None, request_context=None):
        self.attempt_number = attempt_number
        self.operation_model = operation_model
        self.parsed_response = parsed_response
        self.http_response = http_response
        self.caught_exception = caught_exception
        if request_context is None:
            request_context = {}
        self.request_context = request_context
        self._retry_metadata = {}

    def get_error_code(self):
        """Check if there was a parsed response with an error code.

        If we could not find any error codes, ``None`` is returned.

        """
        if self.parsed_response is None:
            return
        error = self.parsed_response.get('Error', {})
        if not isinstance(error, dict):
            return
        return error.get('Code')

    def add_retry_metadata(self, **kwargs):
        """Add key/value pairs to the retry metadata.

        This allows any objects during the retry process to add
        metadata about any checks/validations that happened.

        This gets added to the response metadata in the retry handler.

        """
        self._retry_metadata.update(**kwargs)

    def get_retry_metadata(self):
        return self._retry_metadata.copy()