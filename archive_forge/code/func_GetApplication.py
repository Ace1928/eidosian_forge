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
def GetApplication(client: runapps_v1alpha1_client.RunappsV1alpha1, app_ref: resources) -> Optional[runapps_v1alpha1_messages.Application]:
    """Calls GetApplication API of Runapps for the specified reference.

  Args:
    client: The api client to use.
    app_ref: The resource reference of the application.

  Raises:
    exceptions.UnsupportedIntegrationsLocationError: if the region does not
      exist for the user.

  Returns:
    The application.  If the application does not exist, then
    None is returned.
  """
    request = client.MESSAGES_MODULE.RunappsProjectsLocationsApplicationsGetRequest(name=app_ref.RelativeName())
    try:
        return client.projects_locations_applications.Get(request)
    except apitools_exceptions.HttpNotFoundError:
        return None
    except apitools_exceptions.HttpForbiddenError as e:
        _HandleLocationError(app_ref.locationsId, e)