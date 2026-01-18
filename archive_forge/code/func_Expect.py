import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def Expect(self, request, response=None, exception=None, enable_type_checking=True, **unused_kwargs):
    """Add an expectation on the mocked method.

        Exactly one of response and exception should be specified.

        Args:
          request: The request that should be expected
          response: The response that should be returned or None if
              exception is provided.
          exception: An exception that should be thrown, or None.
          enable_type_checking: When true, the message type of the request
              and response (if provided) will be checked against the types
              required by this method.
        """
    if enable_type_checking:
        self._TypeCheck(request, is_request=True)
        if response:
            self._TypeCheck(response, is_request=False)
    self.__mocked_client._request_responses.append(_ExpectedRequestResponse(self.__key, request, response=response, exception=exception))