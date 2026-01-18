from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetTarget(target_ref):
    """Gets the target message by calling the get target API.

  Args:
    target_ref: protorpc.messages.Message, protorpc.messages.Message, target
      reference.

  Returns:
    Target message.
  Raises:
    Exceptions raised by TargetsClient's get functions
  """
    return target.TargetsClient().Get(target_ref.RelativeName())