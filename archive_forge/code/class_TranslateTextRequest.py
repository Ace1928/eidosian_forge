from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateTextRequest(_messages.Message):
    """The request message for synchronous translation.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.
      See https://cloud.google.com/translate/docs/labels for more information.

  Fields:
    contents: Required. The content of the input in string format. We
      recommend the total content be less than 30k codepoints. The max length
      of this field is 1024. Use BatchTranslateText for larger text.
    glossaryConfig: Optional. Glossary to be applied. The glossary must be
      within the same region (have the same location-id) as the model,
      otherwise an INVALID_ARGUMENT (400) error is returned.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter. See
      https://cloud.google.com/translate/docs/labels for more information.
    mimeType: Optional. The format of the source text, for example,
      "text/html", "text/plain". If left blank, the MIME type defaults to
      "text/html".
    model: Optional. The `model` type requested for this translation. The
      format depends on model type: - AutoML Translation models:
      `projects/{project-number-or-id}/locations/{location-id}/models/{model-
      id}` - General (built-in) models: `projects/{project-number-or-
      id}/locations/{location-id}/models/general/nmt`, For global (non-
      regionalized) requests, use `location-id` `global`. For example,
      `projects/{project-number-or-id}/locations/global/models/general/nmt`.
      If not provided, the default Google model (NMT) will be used
    sourceLanguageCode: Optional. The BCP-47 language code of the input text
      if known, for example, "en-US" or "sr-Latn". Supported language codes
      are listed in Language Support. If the source language isn't specified,
      the API attempts to identify the source language automatically and
      returns the source language within the response.
    targetLanguageCode: Required. The BCP-47 language code to use for
      translation of the input text, set to one of the language codes listed
      in Language Support.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata for the request. Label
    keys and values can be no longer than 63 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. Label values are optional.
    Label keys must start with a letter. See
    https://cloud.google.com/translate/docs/labels for more information.

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
    contents = _messages.StringField(1, repeated=True)
    glossaryConfig = _messages.MessageField('TranslateTextGlossaryConfig', 2)
    labels = _messages.MessageField('LabelsValue', 3)
    mimeType = _messages.StringField(4)
    model = _messages.StringField(5)
    sourceLanguageCode = _messages.StringField(6)
    targetLanguageCode = _messages.StringField(7)