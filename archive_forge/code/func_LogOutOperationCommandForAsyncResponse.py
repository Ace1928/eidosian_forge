from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def LogOutOperationCommandForAsyncResponse(response, args):
    """Reminds user of the command to check operation status.

  Args:
    response: Response from CreateOsPolicyAssignment
    args: gcloud args

  Returns:
    The original response
  """
    if args.async_:
        log.out.Print(GetOperationDescribeCommandFormat(args).format(response.name))
    return response