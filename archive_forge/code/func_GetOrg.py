from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api as crm
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetOrg(self, project):
    ancestry = crm.GetAncestry(project_id=project)
    for resource in ancestry.ancestor:
        resource_type = resource.resourceId.type
        resource_id = resource.resourceId.id
        if resource_type == 'project':
            pass
        if resource_type == 'folder':
            pass
        if resource_type == 'organization':
            return resource_id