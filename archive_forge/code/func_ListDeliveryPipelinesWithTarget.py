from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ListDeliveryPipelinesWithTarget(target_ref, location_ref):
    """Lists the delivery pipelines associated with the specified target.

  The returned list is sorted by the delivery pipeline's create time.
  Args:
    target_ref: protorpc.messages.Message, target object.
    location_ref: protorpc.messages.Message, location object.

  Returns:
    a sorted list of delivery pipelines.
  """
    filter_str = _PIPELINES_WITH_GIVEN_TARGET_FILTER_TEMPLATE.format(target_ref.Name())
    pipelines = delivery_pipeline.DeliveryPipelinesClient().List(location=location_ref.RelativeName(), filter_str=filter_str, page_size=0)
    return sorted(pipelines, key=lambda pipeline: pipeline.createTime, reverse=True)