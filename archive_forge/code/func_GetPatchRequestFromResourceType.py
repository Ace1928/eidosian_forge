from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetPatchRequestFromResourceType(resource_type, name, local_value, etag):
    """Returns the GoogleCloudResourcesettingsV1Setting from the user-specified arguments.

  Args:
    resource_type: A String object that contains the resource type
    name: The resource name of the setting and has the following syntax:
      [organizations|folders|projects]/{resource_id}/settings/{setting_name}.
    local_value: The configured value of the setting at the given parent
      resource
    etag: A fingerprint used for optimistic concurrency.
  """
    setting = settings_service.ResourceSettingsMessages().GoogleCloudResourcesettingsV1Setting(name=name, localValue=local_value, etag=etag)
    if resource_type == ORGANIZATION:
        request = settings_service.ResourceSettingsMessages().ResourcesettingsOrganizationsSettingsPatchRequest(name=name, googleCloudResourcesettingsV1Setting=setting)
    elif resource_type == FOLDER:
        request = settings_service.ResourceSettingsMessages().ResourcesettingsFoldersSettingsPatchRequest(name=name, googleCloudResourcesettingsV1Setting=setting)
    else:
        request = settings_service.ResourceSettingsMessages().ResourcesettingsProjectsSettingsPatchRequest(name=name, googleCloudResourcesettingsV1Setting=setting)
    return request