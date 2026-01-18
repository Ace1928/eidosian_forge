from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1ClientState(_messages.Message):
    """Represents the state associated with an API client calling the Devices
  API. Resource representing ClientState and supports updates from API users

  Enums:
    ComplianceStateValueValuesEnum: The compliance state of the resource as
      specified by the API client.
    HealthScoreValueValuesEnum: The Health score of the resource. The Health
      score is the callers specification of the condition of the device from a
      usability point of view. For example, a third-party device management
      provider may specify a health score based on its compliance with
      organizational policies.
    ManagedValueValuesEnum: The management state of the resource as specified
      by the API client.
    OwnerTypeValueValuesEnum: Output only. The owner of the ClientState

  Messages:
    KeyValuePairsValue: The map of key-value attributes stored by callers
      specific to a device. The total serialized length of this map may not
      exceed 10KB. No limit is placed on the number of attributes in a map.

  Fields:
    assetTags: The caller can specify asset tags for this resource
    complianceState: The compliance state of the resource as specified by the
      API client.
    createTime: Output only. The time the client state data was created.
    customId: This field may be used to store a unique identifier for the API
      resource within which these CustomAttributes are a field.
    etag: The token that needs to be passed back for concurrency control in
      updates. Token needs to be passed back in UpdateRequest
    healthScore: The Health score of the resource. The Health score is the
      callers specification of the condition of the device from a usability
      point of view. For example, a third-party device management provider may
      specify a health score based on its compliance with organizational
      policies.
    keyValuePairs: The map of key-value attributes stored by callers specific
      to a device. The total serialized length of this map may not exceed
      10KB. No limit is placed on the number of attributes in a map.
    lastUpdateTime: Output only. The time the client state data was last
      updated.
    managed: The management state of the resource as specified by the API
      client.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      ClientState in format:
      `devices/{device}/deviceUsers/{device_user}/clientState/{partner}`,
      where partner corresponds to the partner storing the data. For partners
      belonging to the "BeyondCorp Alliance", this is the partner ID specified
      to you by Google. For all other callers, this is a string of the form:
      `{customer}-suffix`, where `customer` is your customer ID. The *suffix*
      is any string the caller specifies. This string will be displayed
      verbatim in the administration console. This suffix is used in setting
      up Custom Access Levels in Context-Aware Access. Your organization's
      customer ID can be obtained from the URL: `GET
      https://www.googleapis.com/admin/directory/v1/customers/my_customer` The
      `id` field in the response contains the customer ID starting with the
      letter 'C'. The customer ID to be used in this API is the string after
      the letter 'C' (not including 'C')
    ownerType: Output only. The owner of the ClientState
    scoreReason: A descriptive cause of the health score.
  """

    class ComplianceStateValueValuesEnum(_messages.Enum):
        """The compliance state of the resource as specified by the API client.

    Values:
      COMPLIANCE_STATE_UNSPECIFIED: The compliance state of the resource is
        unknown or unspecified.
      COMPLIANT: Device is compliant with third party policies
      NON_COMPLIANT: Device is not compliant with third party policies
    """
        COMPLIANCE_STATE_UNSPECIFIED = 0
        COMPLIANT = 1
        NON_COMPLIANT = 2

    class HealthScoreValueValuesEnum(_messages.Enum):
        """The Health score of the resource. The Health score is the callers
    specification of the condition of the device from a usability point of
    view. For example, a third-party device management provider may specify a
    health score based on its compliance with organizational policies.

    Values:
      HEALTH_SCORE_UNSPECIFIED: Default value
      VERY_POOR: The object is in very poor health as defined by the caller.
      POOR: The object is in poor health as defined by the caller.
      NEUTRAL: The object health is neither good nor poor, as defined by the
        caller.
      GOOD: The object is in good health as defined by the caller.
      VERY_GOOD: The object is in very good health as defined by the caller.
    """
        HEALTH_SCORE_UNSPECIFIED = 0
        VERY_POOR = 1
        POOR = 2
        NEUTRAL = 3
        GOOD = 4
        VERY_GOOD = 5

    class ManagedValueValuesEnum(_messages.Enum):
        """The management state of the resource as specified by the API client.

    Values:
      MANAGED_STATE_UNSPECIFIED: The management state of the resource is
        unknown or unspecified.
      MANAGED: The resource is managed.
      UNMANAGED: The resource is not managed.
    """
        MANAGED_STATE_UNSPECIFIED = 0
        MANAGED = 1
        UNMANAGED = 2

    class OwnerTypeValueValuesEnum(_messages.Enum):
        """Output only. The owner of the ClientState

    Values:
      OWNER_TYPE_UNSPECIFIED: Unknown owner type
      OWNER_TYPE_CUSTOMER: Customer is the owner
      OWNER_TYPE_PARTNER: Partner is the owner
    """
        OWNER_TYPE_UNSPECIFIED = 0
        OWNER_TYPE_CUSTOMER = 1
        OWNER_TYPE_PARTNER = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class KeyValuePairsValue(_messages.Message):
        """The map of key-value attributes stored by callers specific to a
    device. The total serialized length of this map may not exceed 10KB. No
    limit is placed on the number of attributes in a map.

    Messages:
      AdditionalProperty: An additional property for a KeyValuePairsValue
        object.

    Fields:
      additionalProperties: Additional properties of type KeyValuePairsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a KeyValuePairsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleAppsCloudidentityDevicesV1CustomAttributeValue
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleAppsCloudidentityDevicesV1CustomAttributeValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    assetTags = _messages.StringField(1, repeated=True)
    complianceState = _messages.EnumField('ComplianceStateValueValuesEnum', 2)
    createTime = _messages.StringField(3)
    customId = _messages.StringField(4)
    etag = _messages.StringField(5)
    healthScore = _messages.EnumField('HealthScoreValueValuesEnum', 6)
    keyValuePairs = _messages.MessageField('KeyValuePairsValue', 7)
    lastUpdateTime = _messages.StringField(8)
    managed = _messages.EnumField('ManagedValueValuesEnum', 9)
    name = _messages.StringField(10)
    ownerType = _messages.EnumField('OwnerTypeValueValuesEnum', 11)
    scoreReason = _messages.StringField(12)