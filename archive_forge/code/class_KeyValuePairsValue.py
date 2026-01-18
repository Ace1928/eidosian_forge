from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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