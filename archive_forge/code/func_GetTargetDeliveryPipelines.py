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
def GetTargetDeliveryPipelines(target_ref):
    """Get all pipelines associated with a target.

  Args:
    target_ref: protorpc.messages.Message, target object.

  Returns:
    A list of delivery pipelines sorted by creation date, or an null list if
    there is an error fetching the pipelines.

  """
    target_dict = target_ref.AsDict()
    location_ref = resources.REGISTRY.Parse(None, collection='clouddeploy.projects.locations', params={'projectsId': target_dict['projectsId'], 'locationsId': target_dict['locationsId']})
    try:
        return delivery_pipeline_util.ListDeliveryPipelinesWithTarget(target_ref, location_ref)
    except apitools_exceptions.HttpError as error:
        log.warning('Failed to fetch pipelines for target: ' + error.content)
        return None