import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def ValidateAndRespond(self, key, request):
    """Validate that key and request match expectations, and respond if so.

        Args:
          key: str, Actual key to compare against expectations.
          request: protorpc.messages.Message or [protorpc.messages.Message]
            or number or string, Actual request to compare againt expectations

        Raises:
          UnexpectedRequestException: If key or request dont match
              expectations.
          apitools_base.Error: If a non-None exception is specified to
              be thrown.

        Returns:
          The response that was specified to be returned.

        """
    if key != self.__key or not (self.__request == request or _MessagesEqual(request, self.__request)):
        raise UnexpectedRequestException((key, request), (self.__key, self.__request))
    if self.__exception:
        raise self.__exception
    return self.__response