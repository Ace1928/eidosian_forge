from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
class HttpExceptionAdditionalHelp(object):
    """Additional help text generator when specific HttpException was raised.

  Attributes:
     known_exc: googlecloudsdk.api_lib.util.exceptions.HttpException, The
      exception to handle.
    error_msg_signature: string, The signature message to determine the
      nature of the error.
    additional_help: string, The additional help to print if error_msg_signature
      appears in the exception error message.
  """

    def __init__(self, known_exc, error_msg_signature, additional_help):
        self.known_exc = known_exc
        self.error_msg_signature = error_msg_signature
        self.additional_help = additional_help

    def Extend(self, msg):
        """Appends the additional help to the given msg."""
        if self.error_msg_signature in self.known_exc.message:
            return '{0}\n\n{1}'.format(msg, console_attr.SafeText(self.additional_help))
        else:
            return msg