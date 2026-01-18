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
def DescribeTarget(target_ref, pipeline_id, skip_pipeline_lookup, list_all_pipelines):
    """Describes details specific to the individual target, delivery pipeline qualified.

  Args:
    target_ref: protorpc.messages.Message, target reference.
    pipeline_id: str, delivery pipeline ID.
    skip_pipeline_lookup: Boolean, flag indicating whether to fetch information
      from the pipeline(s) containing target. If set, pipeline information will
      not be fetched.
    list_all_pipelines: Boolean, flag indicating whether to fetch information
      from all pipelines associated with target, if set to false, it will fetch
      information from the most recently updated one.

  Returns:
    A dictionary of <section name:output>.

  """
    target_obj = target_util.GetTarget(target_ref)
    output = {'Target': target_obj}
    if skip_pipeline_lookup:
        return output
    if pipeline_id:
        return DescribeTargetWithPipeline(target_obj, target_ref, pipeline_id, output)
    else:
        return DescribeTargetWithNoPipeline(target_obj, target_ref, list_all_pipelines, output)