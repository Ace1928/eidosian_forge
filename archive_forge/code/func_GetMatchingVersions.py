from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def GetMatchingVersions(all_versions, versions, service):
    """Return a list of versions to act on based on user arguments.

  Args:
    all_versions: list of Version representing all services in the project.
    versions: list of string, version names to filter for.
      If empty, match all versions.
    service: string or None, service name. If given, only match versions in the
      given service.

  Returns:
    list of matching Version

  Raises:
    VersionValidationError: If an improper combination of arguments is given.
  """
    filtered_versions = all_versions
    if service:
        if service not in [v.service for v in all_versions]:
            raise VersionValidationError('Service [{0}] not found.'.format(service))
        filtered_versions = [v for v in all_versions if v.service == service]
    if versions:
        filtered_versions = [v for v in filtered_versions if v.id in versions]
    return filtered_versions