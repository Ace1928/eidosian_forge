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
def ListPackages(client, messages, repo, page_size=None):
    """Lists all packages under a repository."""
    list_pkgs_req = messages.ArtifactregistryProjectsLocationsRepositoriesPackagesListRequest(parent=repo)
    return list(list_pager.YieldFromList(client.projects_locations_repositories_packages, list_pkgs_req, batch_size=page_size, batch_size_attribute='pageSize', field='packages'))