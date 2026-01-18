from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetServiceFromResourceType(resource_type):
    """Returns the service from the resource type input.

  Args:
    resource_type: A String object that contains the resource type
  """
    if resource_type == ORGANIZATION:
        service = settings_service.OrganizationsSettingsService()
    elif resource_type == FOLDER:
        service = settings_service.FoldersSettingsService()
    else:
        service = settings_service.ProjectsSettingsService()
    return service