from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessRequest(_messages.Message):
    """Request message for the ProcessDocument method.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints) and can only contain lowercase letters, numeric
      characters, underscores, and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.

  Fields:
    fieldMask: Specifies which fields to include in the
      ProcessResponse.document output. Only supports top-level document and
      pages field, so it must be in the form of `{document_field_name}` or
      `pages.{page_field_name}`.
    gcsDocument: A raw document on Google Cloud Storage.
    inlineDocument: An inline document proto.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints) and can only contain lowercase letters, numeric characters,
      underscores, and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter.
    processOptions: Inference-time options for the process API
    rawDocument: A raw document content (bytes).
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
    fieldMask = _messages.StringField(1)
    gcsDocument = _messages.MessageField('GoogleCloudDocumentaiV1GcsDocument', 2)
    inlineDocument = _messages.MessageField('GoogleCloudDocumentaiV1Document', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    processOptions = _messages.MessageField('GoogleCloudDocumentaiV1ProcessOptions', 5)
    rawDocument = _messages.MessageField('GoogleCloudDocumentaiV1RawDocument', 6)
    skipHumanReview = _messages.BooleanField(7)