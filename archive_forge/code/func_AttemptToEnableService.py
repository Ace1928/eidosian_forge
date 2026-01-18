from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.endpoints import config_reporter
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six.moves.urllib.parse
def AttemptToEnableService(self, service_name, is_async):
    """Attempt to enable a service. If lacking permission, log a warning.

    Args:
      service_name: The service to enable.
      is_async: If true, return immediately instead of waiting for the operation
          to finish.
    """
    project_id = properties.VALUES.core.project.Get(required=True)
    try:
        enable_api.EnableService(project_id, service_name, is_async)
        log.status.Print('\n')
    except services_exceptions.EnableServicePermissionDeniedException:
        log.warning('Attempted to enable service [{0}] on project [{1}], but did not have required permissions. Please ensure this service is enabled before using your Endpoints service.\n'.format(service_name, project_id))