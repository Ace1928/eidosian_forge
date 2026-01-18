from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class OldConfigValue(_messages.Message):
    """Optional. Service configuration against which the comparison will be
    done. For this version of API, the supported types are
    google.api.servicemanagement.v1.ConfigRef,
    google.api.servicemanagement.v1.ConfigSource, and google.api.Service

    Messages:
      AdditionalProperty: An additional property for a OldConfigValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a OldConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)