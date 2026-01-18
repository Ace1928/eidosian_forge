from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import operations
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib.projects import util as  projects_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def _GetProjectIds(limit=None):
    """Returns a list of project IDs the current user can list.

  Args:
    limit: int, the maximum number of project ids to return.

  Returns:
    list of str, project IDs, or None (if the command fails).
  """
    try:
        projects = projects_api.List(limit=limit)
        return sorted([project.projectId for project in projects])
    except Exception as err:
        log.warning('Listing available projects failed: %s', six.text_type(err))
        return None