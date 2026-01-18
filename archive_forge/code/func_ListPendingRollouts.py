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
def ListPendingRollouts(target_ref, pipeline_ref):
    """Lists the rollouts in PENDING_APPROVAL state for the releases associated with the specified target.

  The rollouts must be approvalState=NEEDS_APPROVAL and
  state=PENDING_APPROVAL. The returned list is sorted by rollout's create
  time.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_ref: protorpc.messages.Message, pipeline object.

  Returns:
    a sorted list of rollouts.
  """
    filter_str = PENDING_APPROVAL_FILTER_TEMPLATE.format(target_ref.Name())
    parent = WILDCARD_RELEASE_NAME_TEMPLATE.format(pipeline_ref.RelativeName())
    return rollout.RolloutClient().List(release_name=parent, filter_str=filter_str, order_by=PENDING_ROLLOUT_ORDERBY)