from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def DescribeTargetWithPipeline(target_obj, target_ref, pipeline_id, output):
    """Describes details specific to the individual target, delivery pipeline qualified.

  The output contains four sections:

  target
    - detail of the target to be described.

  latest release
    - the detail of the active release in the target.

  latest rollout
    - the detail of the active rollout in the target.

  deployed
    - timestamp of the last successful deployment.

  pending approvals
    - list rollouts that require approval.
  Args:
    target_obj: protorpc.messages.Message, target object.
    target_ref: protorpc.messages.Message, target reference.
    pipeline_id: str, delivery pipeline ID.
    output: A dictionary of <section name:output>.

  Returns:
    A dictionary of <section name:output>.

  """
    target_dict = target_ref.AsDict()
    pipeline_ref = resources.REGISTRY.Parse(None, collection='clouddeploy.projects.locations.deliveryPipelines', params={'projectsId': target_dict['projectsId'], 'locationsId': target_dict['locationsId'], 'deliveryPipelinesId': pipeline_id})
    current_rollout = target_util.GetCurrentRollout(target_ref, pipeline_ref)
    output = SetCurrentReleaseAndRollout(current_rollout, output)
    if target_obj.requireApproval:
        output = ListPendingApprovals(target_ref, pipeline_ref, output)
    return output