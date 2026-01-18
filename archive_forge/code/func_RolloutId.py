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
def RolloutId(rollout_name_or_id):
    """Returns rollout ID.

  Args:
    rollout_name_or_id: str, rollout full name or ID.

  Returns:
    Rollout ID.
  """
    rollout_id = rollout_name_or_id
    if 'projects/' in rollout_name_or_id:
        rollout_id = resources.REGISTRY.ParseRelativeName(rollout_name_or_id, collection=_ROLLOUT_COLLECTION).Name()
    return rollout_id