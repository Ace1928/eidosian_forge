import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def _TypeCheck(self, msg, is_request):
    """Ensure the given message is of the expected type of this method.

        Args:
          msg: The message instance to check.
          is_request: True to validate against the expected request type,
             False to validate against the expected response type.

        Raises:
          exceptions.ConfigurationValueError: If the type of the message was
             not correct.
        """
    if is_request:
        mode = 'request'
        real_type = self.__request_type
    else:
        mode = 'response'
        real_type = self.__response_type
    if not isinstance(msg, real_type):
        raise exceptions.ConfigurationValueError('Expected {} is not of the correct type for method [{}].\n   Required: [{}]\n   Given:    [{}]'.format(mode, self.__key, real_type, type(msg)))