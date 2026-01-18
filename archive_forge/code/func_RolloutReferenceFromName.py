from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def RolloutReferenceFromName(rollout_name):
    """Returns a rollout reference object from a rollout message.

  Args:
    rollout_name: str, full canonical resource name of the rollout

  Returns:
    Rollout reference object
  """
    return resources.REGISTRY.ParseRelativeName(rollout_name, collection=_ROLLOUT_COLLECTION)