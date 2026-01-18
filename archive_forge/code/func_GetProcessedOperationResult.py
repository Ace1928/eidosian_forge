from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def GetProcessedOperationResult(result, is_async=False):
    """Validate and process Operation result message for user display.

  This method checks to make sure the result is of type Operation and
  converts the StartTime field from a UTC timestamp to a local datetime
  string.

  Args:
    result: The message to process (expected to be of type Operation)'
    is_async: If False, the method will block until the operation completes.

  Returns:
    The processed message in Python dict form
  """
    if not result:
        return
    messages = GetMessagesModule()
    RaiseIfResultNotTypeOf(result, messages.Operation)
    result_dict = encoding.MessageToDict(result)
    if not is_async:
        op_name = result_dict['name']
        op_ref = resources.REGISTRY.Parse(op_name, collection='servicemanagement.operations')
        log.status.Print('Waiting for async operation {0} to complete...'.format(op_name))
        result_dict = encoding.MessageToDict(WaitForOperation(op_ref, GetClientInstance()))
    return result_dict