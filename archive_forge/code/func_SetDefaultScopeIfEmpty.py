from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def SetDefaultScopeIfEmpty(unused_ref, args, request):
    """Update the request scope to fall back to core project if not specified.

  Used by Asset Search gcloud `modify_request_hooks`. When --scope flag is not
  specified, it will modify the request.scope to fallback to the core properties
  project.

  Args:
    unused_ref: unused.
    args: The argument namespace.
    request: The request to modify.

  Returns:
    The modified request.
  """
    request.scope = GetDefaultScopeIfEmpty(args)
    return request