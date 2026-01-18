from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DataItem(_messages.Message):
    """A piece of data in a Dataset. Could be an image, a video, a document or
  plain text.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your DataItems. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      DataItem(System labels are excluded). See https://goo.gl/xmQnxf for more
      information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    createTime: Output only. Timestamp when this DataItem was created.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      DataItems. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      DataItem(System labels are excluded). See https://goo.gl/xmQnxf for more
      information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Output only. The resource name of the DataItem.
    payload: Required. The data that the DataItem represents (for example, an
      image or a text snippet). The schema of the payload is stored in the
      parent Dataset's metadata schema's dataItemSchemaUri field.
    updateTime: Output only. Timestamp when this DataItem was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    DataItems. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    No more than 64 user labels can be associated with one DataItem(System
    labels are excluded). See https://goo.gl/xmQnxf for more information and
    examples of labels. System reserved label keys are prefixed with
    "aiplatform.googleapis.com/" and are immutable.

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
    etag = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    payload = _messages.MessageField('extra_types.JsonValue', 5)
    updateTime = _messages.StringField(6)