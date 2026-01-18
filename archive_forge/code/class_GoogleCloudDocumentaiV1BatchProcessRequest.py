from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1BatchProcessRequest(_messages.Message):
    """Request message for BatchProcessDocuments.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints) and can only contain lowercase letters, numeric
      characters, underscores, and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.

  Fields:
    documentOutputConfig: The output configuration for the
      BatchProcessDocuments method.
    inputDocuments: The input documents for the BatchProcessDocuments method.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints) and can only contain lowercase letters, numeric characters,
      underscores, and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter.
    processOptions: Inference-time options for the process API
    skipHumanReview: Whether human review should be skipped for this request.
      Default to `false`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata for the request. Label
    keys and values can be no longer than 63 characters (Unicode codepoints)
    and can only contain lowercase letters, numeric characters, underscores,
    and dashes. International characters are allowed. Label values are
    optional. Label keys must start with a letter.

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
    documentOutputConfig = _messages.MessageField('GoogleCloudDocumentaiV1DocumentOutputConfig', 1)
    inputDocuments = _messages.MessageField('GoogleCloudDocumentaiV1BatchDocumentsInputConfig', 2)
    labels = _messages.MessageField('LabelsValue', 3)
    processOptions = _messages.MessageField('GoogleCloudDocumentaiV1ProcessOptions', 4)
    skipHumanReview = _messages.BooleanField(5)