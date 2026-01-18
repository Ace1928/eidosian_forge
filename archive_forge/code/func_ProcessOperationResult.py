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
def ProcessOperationResult(result, is_async=False):
    """Validate and process Operation outcome for user display.

  Args:
    result: The message to process (expected to be of type Operation)'
    is_async: If False, the method will block until the operation completes.

  Returns:
    The processed Operation message in Python dict form
  """
    op = GetProcessedOperationResult(result, is_async)
    if is_async:
        cmd = OP_WAIT_CMD.format(op.get('name'))
        log.status.Print('Asynchronous operation is in progress... Use the following command to wait for its completion:\n {0}\n'.format(cmd))
    else:
        cmd = OP_DESCRIBE_CMD.format(op.get('name'))
        log.status.Print('Operation finished successfully. The following command can describe the Operation details:\n {0}\n'.format(cmd))
    return op