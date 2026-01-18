from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.resource_manager.exceptions import ResourceManagerError
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.core import exceptions as core_exceptions
def _GetGceInstanceCanonicalName(project_identifier, instance_identifier, location, release_track):
    """Returns the correct canonical name for the given gce compute instance.

  Args:
    project_identifier: project number of the compute instance
    instance_identifier: name of the instance
    location: location in which the resource lives
    release_track: release stage of current endpoint

  Returns:
    instance_id: returns the canonical instance id
  """
    compute_holder = base_classes.ComputeApiHolder(release_track)
    client = compute_holder.client
    request = (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(instance=instance_identifier, project=project_identifier, zone=location))
    errors_to_collect = []
    instances = client.MakeRequests([request], errors_to_collect=errors_to_collect)
    if errors_to_collect:
        raise core_exceptions.MultiError(errors_to_collect)
    return str(instances[0].id)