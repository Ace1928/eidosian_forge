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
def TestRedirectionIAMPermission(project):
    """Tests the user has the storage.buckets.update IAM permission on the project."""
    project_ref = project_util.ParseProject(project)
    result = projects_api.TestIamPermissions(project_ref, REDIRECT_PERMISSIONS)
    return set(REDIRECT_PERMISSIONS) == set(result.permissions)