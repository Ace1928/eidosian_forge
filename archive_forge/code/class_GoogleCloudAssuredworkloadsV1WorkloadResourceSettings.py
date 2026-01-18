from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1WorkloadResourceSettings(_messages.Message):
    """Represent the custom settings for the resources to be created.

  Enums:
    ResourceTypeValueValuesEnum: Indicates the type of resource. This field
      should be specified to correspond the id to the right project type
      (CONSUMER_PROJECT or ENCRYPTION_KEYS_PROJECT)

  Fields:
    displayName: User-assigned resource display name. If not empty it will be
      used to create a resource with the specified name.
    resourceId: Resource identifier. For a project this represents project_id.
      If the project is already taken, the workload creation will fail. For
      KeyRing, this represents the keyring_id. For a folder, don't set this
      value as folder_id is assigned by Google.
    resourceType: Indicates the type of resource. This field should be
      specified to correspond the id to the right project type
      (CONSUMER_PROJECT or ENCRYPTION_KEYS_PROJECT)
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """Indicates the type of resource. This field should be specified to
    correspond the id to the right project type (CONSUMER_PROJECT or
    ENCRYPTION_KEYS_PROJECT)

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Unknown resource type.
      CONSUMER_PROJECT: Deprecated. Existing workloads will continue to
        support this, but new CreateWorkloadRequests should not specify this
        as an input value.
      CONSUMER_FOLDER: Consumer Folder.
      ENCRYPTION_KEYS_PROJECT: Consumer project containing encryption keys.
      KEYRING: Keyring resource that hosts encryption keys.
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        CONSUMER_PROJECT = 1
        CONSUMER_FOLDER = 2
        ENCRYPTION_KEYS_PROJECT = 3
        KEYRING = 4
    displayName = _messages.StringField(1)
    resourceId = _messages.StringField(2)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 3)