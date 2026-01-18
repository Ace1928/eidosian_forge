from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceResourceBindingConfig(_messages.Message):
    """Message for a resource binding, defined in config of the source
  resource.

  Messages:
    BindingConfigValue: Any configs associated with the binding. Supported
      keys are resource type specific.

  Fields:
    binding_config: Any configs associated with the binding. Supported keys
      are resource type specific.
    ref: Reference to a target resource that is being bound. Format: "/", e.g.
      "cloudsql/sql_db"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BindingConfigValue(_messages.Message):
        """Any configs associated with the binding. Supported keys are resource
    type specific.

    Messages:
      AdditionalProperty: An additional property for a BindingConfigValue
        object.

    Fields:
      additionalProperties: Additional properties of type BindingConfigValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BindingConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    binding_config = _messages.MessageField('BindingConfigValue', 1)
    ref = _messages.StringField(2)