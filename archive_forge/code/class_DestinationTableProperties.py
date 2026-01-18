from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationTableProperties(_messages.Message):
    """Properties for the destination table.

  Messages:
    LabelsValue: Optional. The labels associated with this table. You can use
      these to organize and group your tables. This will only be used if the
      destination table is newly created. If the table already exists and
      labels are different than the current labels are provided, the job will
      fail.

  Fields:
    description: Optional. The description for the destination table. This
      will only be used if the destination table is newly created. If the
      table already exists and a value different than the current description
      is provided, the job will fail.
    expirationTime: Internal use only.
    friendlyName: Optional. Friendly name for the destination table. If the
      table already exists, it should be same as the existing friendly name.
    labels: Optional. The labels associated with this table. You can use these
      to organize and group your tables. This will only be used if the
      destination table is newly created. If the table already exists and
      labels are different than the current labels are provided, the job will
      fail.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels associated with this table. You can use these to
    organize and group your tables. This will only be used if the destination
    table is newly created. If the table already exists and labels are
    different than the current labels are provided, the job will fail.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    expirationTime = _message_types.DateTimeField(2)
    friendlyName = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)