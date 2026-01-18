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
def ListAllPipelineReleaseRollout(target_ref, pipeline_refs):
    """Retrieves latest rollout and release information for each delivery pipeline.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_refs: protorpc.messages.Message a list of DeliveryPipeline objects

  Returns:
    A content directory.

  """
    output = []
    for pipeline_ref in pipeline_refs:
        pipeline_entry = SetPipelineReleaseRollout(target_ref, pipeline_ref)
        output.append({pipeline_ref.RelativeName(): pipeline_entry})
    return output