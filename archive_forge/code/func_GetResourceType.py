from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def GetResourceType(key, release_track):
    """Returns a resource settings type from the key.

  Args:
    key: str, the setting name, which can be one of the following -
      consumer-project-id, consumer-project-name, encryption-keys-project-id,
      encryption-keys-project-name or keyring-id.
    release_track: ReleaseTrack, gcloud release track being used.
  """
    resource_settings_message = GetResourceSettings(release_track)
    if key.startswith('consumer-project'):
        return resource_settings_message.ResourceTypeValueValuesEnum.CONSUMER_PROJECT
    elif key.startswith('encryption-keys-project'):
        return resource_settings_message.ResourceTypeValueValuesEnum.ENCRYPTION_KEYS_PROJECT
    elif key.startswith('keyring'):
        return resource_settings_message.ResourceTypeValueValuesEnum.KEYRING