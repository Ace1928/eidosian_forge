from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValuedResource(_messages.Message):
    """A resource that is determined to have value to a user's system

  Enums:
    ResourceValueValueValuesEnum: How valuable this resource is.

  Fields:
    displayName: Human-readable name of the valued resource.
    exposedScore: Exposed score for this valued resource. A value of 0 means
      no exposure was detected exposure.
    name: Valued resource name, for example, e.g.:
      `organizations/123/simulations/456/valuedResources/789`
    resource: The [full resource name](https://cloud.google.com/apis/design/re
      source_names#full_resource_name) of the valued resource.
    resourceType: The [resource type](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types) of the valued resource.
    resourceValue: How valuable this resource is.
    resourceValueConfigsUsed: List of resource value configurations' metadata
      used to determine the value of this resource. Maximum of 100.
  """

    class ResourceValueValueValuesEnum(_messages.Enum):
        """How valuable this resource is.

    Values:
      RESOURCE_VALUE_UNSPECIFIED: The resource value isn't specified.
      RESOURCE_VALUE_LOW: This is a low-value resource.
      RESOURCE_VALUE_MEDIUM: This is a medium-value resource.
      RESOURCE_VALUE_HIGH: This is a high-value resource.
    """
        RESOURCE_VALUE_UNSPECIFIED = 0
        RESOURCE_VALUE_LOW = 1
        RESOURCE_VALUE_MEDIUM = 2
        RESOURCE_VALUE_HIGH = 3
    displayName = _messages.StringField(1)
    exposedScore = _messages.FloatField(2)
    name = _messages.StringField(3)
    resource = _messages.StringField(4)
    resourceType = _messages.StringField(5)
    resourceValue = _messages.EnumField('ResourceValueValueValuesEnum', 6)
    resourceValueConfigsUsed = _messages.MessageField('ResourceValueConfigMetadata', 7, repeated=True)