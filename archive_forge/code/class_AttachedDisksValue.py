from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AttachedDisksValue(_messages.Message):
    """Status of attached disks.

    Messages:
      AdditionalProperty: An additional property for a AttachedDisksValue
        object.

    Fields:
      additionalProperties: Additional properties of type AttachedDisksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AttachedDisksValue object.

      Fields:
        key: Name of the additional property.
        value: A DiskStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('DiskStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)