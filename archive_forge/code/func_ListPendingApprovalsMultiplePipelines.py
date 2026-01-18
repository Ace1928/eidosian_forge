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
def ListPendingApprovalsMultiplePipelines(target_ref, pipeline_refs, output):
    """Fetches a list of pending rollouts for each pipeline and appends the result to a single list.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_refs: protorpc.messages.Message, list of pipeline objects.
    output: dictionary object

  Returns:
    The modified output object with the list of pending rollouts.

  """
    all_pending_approvals = []
    for pipeline_ref in pipeline_refs:
        result_dict = ListPendingApprovals(target_ref, pipeline_ref, {})
        approvals_one_pipeline = result_dict.get('Pending Approvals', [])
        all_pending_approvals.extend(approvals_one_pipeline)
    output['Pending Approvals'] = all_pending_approvals
    return output