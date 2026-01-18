from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HistoricalEvent(_messages.Message):
    """Message describing HistoricalEvent object

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    ipRange: Required. IP range of registry node to view history.
    labels: Optional. Labels as key value pairs
    name: Required. name of resource
    nodeEvents: Output only. List of node events of IP range.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    createTime = _messages.StringField(1)
    ipRange = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    nodeEvents = _messages.MessageField('NodeEvent', 5, repeated=True)
    updateTime = _messages.StringField(6)