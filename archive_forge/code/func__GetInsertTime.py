from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import enum
from googlecloudsdk.api_lib.app import exceptions as app_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def _GetInsertTime(operation):
    """Finds the insertTime property and return its string form.

  Args:
    operation: A messages.Operation instance.

  Returns:
    The time the operation started in string form or None if N/A.
  """
    if not operation.metadata:
        return None
    properties = operation.metadata.additionalProperties
    for prop in properties:
        if prop.key == 'insertTime':
            return prop.value.string_value