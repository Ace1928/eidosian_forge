from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from typing import List, Optional
from apitools.base.py import encoding as apitools_encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import retry
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetApplicationStatus(client: runapps_v1alpha1_client.RunappsV1alpha1, app_ref: resources, resource_ids: Optional[List[runapps_v1alpha1_messages.ResourceID]]=None) -> Optional[runapps_v1alpha1_messages.ApplicationStatus]:
    """Calls GetApplicationStatus API of Runapps for the specified reference.

  Args:
    client: the api client to use.
    app_ref: the resource reference of the application.
    resource_ids: ResourceID of the resource to get status for. If not given,
      all resources in the application will be queried.

  Returns:
    The ApplicationStatus object. Or None if not found.
  """
    if resource_ids:
        res_filters = [res_id.type + '/' + res_id.name for res_id in resource_ids]
    else:
        res_filters = []
    module = client.MESSAGES_MODULE
    request = module.RunappsProjectsLocationsApplicationsGetStatusRequest(name=app_ref.RelativeName(), resources=res_filters)
    try:
        return client.projects_locations_applications.GetStatus(request)
    except apitools_exceptions.HttpNotFoundError:
        return None