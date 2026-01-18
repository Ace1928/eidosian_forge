from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamContent(_messages.Message):
    """Message describing StreamContent object Next ID: 10

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    bucketName: Name of the Cloud Storage bucket in the consumer project that
      stores the content source.
    buildVersions: Output only. User-specified version tags and unique build
      IDs of content builds
    contentVersionTags: Output only. User-specified version tags of content
      builds
    createTime: Output only. [Output only] Create time stamp
    labels: Labels as key value pairs
    lifecycleState: Output only. Current state of the content.
    name: Identifier. name of resource
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

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
    bucketName = _messages.StringField(1)
    buildVersions = _messages.MessageField('BuildVersion', 2, repeated=True)
    contentVersionTags = _messages.StringField(3, repeated=True)
    createTime = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    lifecycleState = _messages.MessageField('LifecycleState', 6)
    name = _messages.StringField(7)
    updateTime = _messages.StringField(8)