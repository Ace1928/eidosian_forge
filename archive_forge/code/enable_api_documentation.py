from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.core import log
Check to see if the service is enabled, and if it is not, do so.

  Args:
    project_id: The ID of the project for which to enable the service.
    service_name: The name of the service to enable on the project.
    is_async: bool, if True, print the operation ID and return immediately,
           without waiting for the op to complete.

  Raises:
    exceptions.ListServicesPermissionDeniedException: if a 403 or 404 error
        is returned by the listing service.
    exceptions.EnableServicePermissionDeniedException: when enabling the API
        fails with a 403 or 404 error code.
    api_lib_exceptions.HttpException: Another miscellaneous error with the
        servicemanagement service.
  