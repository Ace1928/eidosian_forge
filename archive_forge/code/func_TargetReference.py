from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def TargetReference(target_name_or_id, project, location_id):
    """Creates the target reference base on the parameters.

  Returns the shared target reference.

  Args:
    target_name_or_id: str, target full name or ID.
    project: str,project number or ID.
    location_id: str, region ID.

  Returns:
    Target reference.
  """
    return resources.REGISTRY.Parse(None, collection=_SHARED_TARGET_COLLECTION, params={'projectsId': project, 'locationsId': location_id, 'targetsId': TargetId(target_name_or_id)})