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
def DescribeTargetWithNoPipeline(target_obj, target_ref, list_all_pipelines, output):
    """Describes details specific to the individual target.

  In addition, it will also display details about pipelines associated
  with the given target.

  The output contains the following sections:

  target
    - details of the target to be described.

  associated pipelines
    - details of the pipelines that use the target.

  For each associated pipeline, the following will be displayed:

  latest release
    - details of the active release in the target.

  latest rollout
    - details of the active rollout in the target.

  deployed
    - timestamp of the last successful deployment.

  pending approvals
    - list the rollouts that require approval.

  Args:
    target_obj: protorpc.messages.Message, target object.
    target_ref: protorpc.messages.Message, target reference.
    list_all_pipelines: Boolean, if true, will return information about all
      pipelines associated with target, otherwise, the most recently active
      pipeline information will be displayed.
    output: A dictionary of <section name:output>.

  Returns:
    A dictionary of <section name:output>.

  """
    sorted_pipelines = GetTargetDeliveryPipelines(target_ref)
    if not sorted_pipelines:
        return output
    output['Number of associated delivery pipelines'] = len(sorted_pipelines)
    pipeline_refs = list(map(delivery_pipeline_util.PipelineToPipelineRef, sorted_pipelines))
    if list_all_pipelines:
        output['Associated delivery pipelines'] = ListAllPipelineReleaseRollout(target_ref, pipeline_refs)
        if target_obj.requireApproval:
            ListPendingApprovalsMultiplePipelines(target_ref, pipeline_refs, output)
    else:
        active_pipeline_ref, latest_rollout = GetMostRecentlyActivePipeline(target_ref, pipeline_refs)
        if active_pipeline_ref and latest_rollout:
            output['Active Pipeline'] = SetMostRecentlyActivePipeline(active_pipeline_ref, latest_rollout)
        if target_obj.requireApproval:
            ListPendingApprovals(target_ref, active_pipeline_ref, output)
    return output