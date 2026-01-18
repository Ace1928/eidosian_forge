from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import exceptions
def ValidateInstanceOsPolicyAssignment(value, param_name):
    """Check if os policy assignment id is non-null/empty; doesn't check whether it exists.

  Args:
    value: str, the os policy assignment id value to check
    param_name: str, the parameter's name; included in the exception's message

  Raises:
    exceptions.Error: if value is empty
  """
    if not value:
        raise exceptions.Error('Missing required parameter ' + param_name)