from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import resources
def ListVersions(client, messages, pkg, version_view, page_size=None, order_by=None, limit=None):
    """Lists all versions under a package."""
    page_limit = limit
    if limit is None or (page_size is not None and page_size < limit):
        page_limit = page_size
    list_vers_req = messages.ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsListRequest(parent=pkg, view=version_view, orderBy=order_by)
    return list(list_pager.YieldFromList(client.projects_locations_repositories_packages_versions, list_vers_req, limit=limit, batch_size=page_limit, batch_size_attribute='pageSize', field='versions'))