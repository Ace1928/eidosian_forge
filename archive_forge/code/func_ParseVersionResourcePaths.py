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
def ParseVersionResourcePaths(paths, project):
    """Parse the list of resource paths specifying versions.

  Args:
    paths: The list of resource paths by which to filter.
    project: The current project. Used for validation.

  Returns:
    list of Version

  Raises:
    VersionValidationError: If not all versions are valid resource paths for the
      current project.
  """
    versions = list(map(Version.FromResourcePath, paths))
    for version in versions:
        if not (version.project or version.service):
            raise VersionValidationError('If you provide a resource path as an argument, all arguments must be resource paths.')
        if version.project and version.project != project:
            raise VersionValidationError('All versions must be in the current project.')
        version.project = project
    return versions