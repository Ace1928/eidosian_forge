from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AsyncSecondaryDisksValue(_messages.Message):
    """Key: disk, value: AsyncReplicationStatus message

    Messages:
      AdditionalProperty: An additional property for a
        AsyncSecondaryDisksValue object.

    Fields:
      additionalProperties: Additional properties of type
        AsyncSecondaryDisksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AsyncSecondaryDisksValue object.

      Fields:
        key: Name of the additional property.
        value: A DiskResourceStatusAsyncReplicationStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('DiskResourceStatusAsyncReplicationStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)