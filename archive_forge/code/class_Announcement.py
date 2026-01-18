from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Announcement(_messages.Message):
    """Announcement for the resources of Vmware Engine.

  Enums:
    StateValueValuesEnum: Output only. State of the resource. New values may
      be added to this enum when appropriate.

  Messages:
    MetadataValue: Output only. Additional structured details about this
      announcement.

  Fields:
    activityType: Optional. Activity type of the announcement There can be
      only one active announcement for a given activity type and target
      resource.
    code: Required. Code of the announcement. Indicates the presence of a
      VMware Engine related announcement and corresponds to a related message
      in the `description` field.
    createTime: Output only. Creation time of this resource. It also serves as
      start time of notification.
    description: Output only. Description of the announcement.
    metadata: Output only. Additional structured details about this
      announcement.
    name: Output only. The resource name of the announcement. Resource names
      are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-west1-a/announcements/my-announcement-
      id`
    privateCloud: A Private Cloud full resource name.
    state: Output only. State of the resource. New values may be added to this
      enum when appropriate.
    targetResourceType: Output only. Target Resource Type defines the type of
      the target for the announcement
    updateTime: Output only. Last update time of this resource.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the resource. New values may be added to this
    enum when appropriate.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: Active announcement which should be visible to user.
      INACTIVE: Inactive announcement which should not be visible to user.
      DELETING: Announcement which is being deleted
      CREATING: Announcement which being created
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        DELETING = 3
        CREATING = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Output only. Additional structured details about this announcement.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    activityType = _messages.StringField(1)
    code = _messages.StringField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    metadata = _messages.MessageField('MetadataValue', 5)
    name = _messages.StringField(6)
    privateCloud = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    targetResourceType = _messages.StringField(9)
    updateTime = _messages.StringField(10)