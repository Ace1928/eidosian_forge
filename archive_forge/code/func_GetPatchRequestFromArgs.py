from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetPatchRequestFromArgs(args, name, local_value, etag):
    """Returns the GoogleCloudResourcesettingsV1Setting from the user-specified arguments.

  Args:
    resource_type: A String object that contains the resource type
    name: The resource name of the setting and has the following syntax:
      [organizations|folders|projects]/{resource_id}/settings/{setting_name}.
    local_value: The configured value of the setting at the given parent
      resource
    etag: A fingerprint used for optimistic concurrency.
  """
    resource_type = ComputeResourceType(args)
    return GetPatchRequestFromResourceType(resource_type, name, local_value, etag)