from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util
from googlecloudsdk.command_lib.container.fleet.memberships import errors
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import properties
def GetAnthosSupportUser(self, project_id):
    """Get P4SA account name for Anthos Support when user not specified.

    Args:
      project_id: the project ID of the resource.

    Returns:
      the P4SA account name for Anthos Support.
    """
    project_number = projects_api.Get(projects_util.ParseProject(project_id)).projectNumber
    hub_endpoint_override = util.APIEndpoint()
    if hub_endpoint_override == util.PROD_API:
        return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='')
    elif hub_endpoint_override == util.STAGING_API:
        return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='staging-')
    elif hub_endpoint_override == util.AUTOPUSH_API:
        return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='autopush-')
    else:
        raise errors.UnknownApiEndpointOverrideError('gkehub')