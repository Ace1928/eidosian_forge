from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProcessSummary(_messages.Message):
    """Process Summary

  Messages:
    ProcessLogsValue: A ProcessLogsValue object.

  Fields:
    addTime: A string attribute.
    hostPort: A string attribute.
    isActive: A boolean attribute.
    processId: A string attribute.
    processLogs: A ProcessLogsValue attribute.
    removeTime: A string attribute.
    totalCores: A integer attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProcessLogsValue(_messages.Message):
        """A ProcessLogsValue object.

    Messages:
      AdditionalProperty: An additional property for a ProcessLogsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ProcessLogsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProcessLogsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    addTime = _messages.StringField(1)
    hostPort = _messages.StringField(2)
    isActive = _messages.BooleanField(3)
    processId = _messages.StringField(4)
    processLogs = _messages.MessageField('ProcessLogsValue', 5)
    removeTime = _messages.StringField(6)
    totalCores = _messages.IntegerField(7, variant=_messages.Variant.INT32)