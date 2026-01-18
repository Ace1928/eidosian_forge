from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SystemLabelsValue(_messages.Message):
    """Output only. Values for predefined system metadata labels. System
    labels are a kind of metadata extracted by Google, including
    "machine_image", "vpc", "subnet_id", "security_group", "name", etc. System
    label values can be only strings, Boolean values, or a list of strings.
    For example: { "name": "my-test-instance", "security_group": ["a", "b",
    "c"], "spot_instance": false }

    Messages:
      AdditionalProperty: An additional property for a SystemLabelsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SystemLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)