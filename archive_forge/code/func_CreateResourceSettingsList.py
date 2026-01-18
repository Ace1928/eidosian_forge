from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateResourceSettingsList(resource_settings, release_track):
    """Construct a list of ResourceSettings for Assured Workload object.

  Args:
    resource_settings: a list of key=value pairs of customized resource
      settings.
    release_track: ReleaseTrack, gcloud release track being used.

  Returns:
    A list of ResourceSettings for the Assured Workload object.
  """
    resource_settings_dict = {}
    for key, value in resource_settings.items():
        resource_type = GetResourceType(key, release_track)
        resource_settings = resource_settings_dict[resource_type] if resource_type in resource_settings_dict else CreateResourceSettings(resource_type, release_track)
        if key.endswith('-id'):
            resource_settings.resourceId = value
        elif key.endswith('-name'):
            resource_settings.displayName = value
        resource_settings_dict[resource_type] = resource_settings
    return list(resource_settings_dict.values())