from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetErrorCodeAndMessage(error):
    """Returns the individual error code and message from a JSON http response.

  Prefer using GetError(error) unless you need to examine the error code and
  take specific action on it.

  Args:
    error: the Http error response, whose content is a JSON-format string.

  Returns:
    (code, msg) A tuple holding the error code and error message string.

  Raises:
    ValueError: if the error is not in JSON format.
  """
    data = json.loads(error.content)
    return (data['error']['code'], data['error']['message'])