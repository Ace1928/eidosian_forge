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
def CreateDeployment(client: runapps_v1alpha1_client.RunappsV1alpha1, app_ref: resources, deployment: runapps_v1alpha1_messages.Deployment, validate_only: Optional[bool]=False) -> runapps_v1alpha1_messages.Operation:
    """Calls CreateDeployment API of Runapps.

  Args:
    client: the api client to use.
    app_ref: the resource reference of the application the deployment belongs to
    deployment: the deployment object
    validate_only: whether to only validate the deployment

  Returns:
    the LRO of this request.
  """
    return client.projects_locations_applications_deployments.Create(client.MESSAGES_MODULE.RunappsProjectsLocationsApplicationsDeploymentsCreateRequest(parent=app_ref.RelativeName(), deployment=deployment, deploymentId=deployment.name, validateOnly=validate_only))