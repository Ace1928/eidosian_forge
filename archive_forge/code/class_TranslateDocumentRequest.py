from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateDocumentRequest(_messages.Message):
    """A document translation request.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.
      See https://cloud.google.com/translate/docs/advanced/labels for more
      information.

  Fields:
    customizedAttribution: Optional. This flag is to support user customized
      attribution. If not provided, the default is `Machine Translated by
      Google`. Customized attribution should follow rules in
      https://cloud.google.com/translate/attribution#attribution_and_logos
    documentInputConfig: Required. Input configurations.
    documentOutputConfig: Optional. Output configurations. Defines if the
      output file should be stored within Cloud Storage as well as the desired
      output format. If not provided the translated file will only be returned
      through a byte-stream and its output mime type will be the same as the
      input file's mime type.
    enableRotationCorrection: Optional. If true, enable auto rotation
      correction in DVS.
    enableShadowRemovalNativePdf: Optional. If true, use the text removal
      server to remove the shadow text on background image for native pdf
      translation. Shadow removal feature can only be enabled when
      is_translate_native_pdf_only: false && pdf_native_only: false
    glossaryConfig: Optional. Glossary to be applied. The glossary must be
      within the same region (have the same location-id) as the model,
      otherwise an INVALID_ARGUMENT (400) error is returned.
    isTranslateNativePdfOnly: Optional. is_translate_native_pdf_only field for
      external customers. If true, the page limit of online native pdf
      translation is 300 and only native pdf pages will be translated.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter. See
      https://cloud.google.com/translate/docs/advanced/labels for more
      information.
    model: Optional. The `model` type requested for this translation. The
      format depends on model type: - AutoML Translation models:
      `projects/{project-number-or-id}/locations/{location-id}/models/{model-
      id}` - General (built-in) models: `projects/{project-number-or-
      id}/locations/{location-id}/models/general/nmt`, If not provided, the
      default Google model (NMT) will be used for translation.
    sourceLanguageCode: Optional. The BCP-47 language code of the input
      document if known, for example, "en-US" or "sr-Latn". Supported language
      codes are listed in Language Support. If the source language isn't
      specified, the API attempts to identify the source language
      automatically and returns the source language within the response.
      Source language must be specified if the request contains a glossary or
      a custom model.
    targetLanguageCode: Required. The BCP-47 language code to use for
      translation of the input document, set to one of the language codes
      listed in Language Support.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata for the request. Label
    keys and values can be no longer than 63 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. Label values are optional.
    Label keys must start with a letter. See
    https://cloud.google.com/translate/docs/advanced/labels for more
    information.

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
    customizedAttribution = _messages.StringField(1)
    documentInputConfig = _messages.MessageField('DocumentInputConfig', 2)
    documentOutputConfig = _messages.MessageField('DocumentOutputConfig', 3)
    enableRotationCorrection = _messages.BooleanField(4)
    enableShadowRemovalNativePdf = _messages.BooleanField(5)
    glossaryConfig = _messages.MessageField('TranslateTextGlossaryConfig', 6)
    isTranslateNativePdfOnly = _messages.BooleanField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    model = _messages.StringField(9)
    sourceLanguageCode = _messages.StringField(10)
    targetLanguageCode = _messages.StringField(11)