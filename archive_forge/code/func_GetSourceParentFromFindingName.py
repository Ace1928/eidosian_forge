from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetSourceParentFromFindingName(resource_name, version):
    """Get parent (with source) from Finding name i.e remove 'findings/{finding_name}'.

  Args:
    resource_name: finding name {parent with source}/findings/{findingID}
    version: API version.

  Returns:
    The parent with source or parent with source and location
    examples:
    if no location is specified the result will be one of the following forms
      `organizations/{organization_id}/sources/{source_id}`
      `folders/{folders_id}/sources/{source_id}`
      `projects/{projects_id}/sources/{source_id}`
    if a location is specified the result will be one of the following forms
      `organizations/{organization_id}/sources/{source_id}/locations/{location_id}`
      `folders/{folders_id}/sources/{source_id}/locations/{location_id}`
      `projects/{projects_id}/sources/{source_id}/locations/{location_id}`
  """
    resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9]+')
    if not resource_pattern.match(resource_name):
        raise errors.InvalidSCCInputError('When providing a full resource path, it must also include the organization, project, or folder prefix.')
    list_source_components = resource_name.split('/')
    if version == 'v1':
        return f'{GetParentFromResourceName(resource_name)}/{list_source_components[2]}/{list_source_components[3]}'
    if version == 'v2':
        return f'{GetParentFromResourceName(resource_name)}/{list_source_components[2]}/{list_source_components[3]}/{list_source_components[4]}/{list_source_components[5]}'