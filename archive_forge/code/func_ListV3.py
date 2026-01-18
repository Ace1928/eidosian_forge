from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
def ListV3(limit=None, batch_size=500, parent=None):
    """Make API calls to List active projects.

  Args:
    limit: The number of projects to limit the results to. This limit is passed
      to the server and the server does the limiting.
    batch_size: the number of projects to get with each request.
    parent: The parent folder or organization whose children are to be listed.

  Returns:
    Generator that yields projects
  """
    client = projects_util.GetClient('v3')
    messages = projects_util.GetMessages('v3')
    return list_pager.YieldFromList(client.projects, messages.CloudresourcemanagerProjectsListRequest(parent=parent), batch_size=batch_size, limit=limit, field='projects', batch_size_attribute='pageSize')