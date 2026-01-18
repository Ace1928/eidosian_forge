from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegistryNode(_messages.Message):
    """Message describing RegistryNode object

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    attribute: Optional. Attributes represent the custom metadata for registry
      node.
    createTime: Output only. [Output only] Create time stamp
    importInfo: Optional. Metadata contains resource importing information.
    ipRange: Required. IP range of registry node.
    labels: Optional. Labels as key value pairs
    name: Required. Identifier. name of resource
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
    attribute = _messages.MessageField('Attribute', 1, repeated=True)
    createTime = _messages.StringField(2)
    importInfo = _messages.MessageField('ImportInfo', 3)
    ipRange = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)