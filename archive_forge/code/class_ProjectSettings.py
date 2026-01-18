from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectSettings(_messages.Message):
    """Settings that control how a consumer project uses a service.

  Messages:
    PropertiesValue: Service-defined per-consumer properties.  A key-value
      mapping a string key to a google.protobuf.ListValue proto. Values in the
      list are typed as defined in the Service configuration's
      consumer.properties field.

  Fields:
    consumerProjectId: ID for the project consuming this service.
    operations: Read-only view of pending operations affecting this resource,
      if requested.
    properties: Service-defined per-consumer properties.  A key-value mapping
      a string key to a google.protobuf.ListValue proto. Values in the list
      are typed as defined in the Service configuration's consumer.properties
      field.
    quotaSettings: Settings that control how much or how fast the service can
      be used by the consumer project.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.
    usageSettings: Settings that control whether this service is usable by the
      consumer project.
    visibilitySettings: Settings that control which features of the service
      are visible to the consumer project.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Service-defined per-consumer properties.  A key-value mapping a string
    key to a google.protobuf.ListValue proto. Values in the list are typed as
    defined in the Service configuration's consumer.properties field.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2, repeated=True)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consumerProjectId = _messages.StringField(1)
    operations = _messages.MessageField('Operation', 2, repeated=True)
    properties = _messages.MessageField('PropertiesValue', 3)
    quotaSettings = _messages.MessageField('QuotaSettings', 4)
    serviceName = _messages.StringField(5)
    usageSettings = _messages.MessageField('UsageSettings', 6)
    visibilitySettings = _messages.MessageField('VisibilitySettings', 7)