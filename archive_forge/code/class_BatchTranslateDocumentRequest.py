from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchTranslateDocumentRequest(_messages.Message):
    """The BatchTranslateDocument request.

  Messages:
    FormatConversionsValue: Optional. File format conversion map to be applied
      to all input files. Map's key is the original mime_type. Map's value is
      the target mime_type of translated documents. Supported file format
      conversion includes: - `application/pdf` to
      `application/vnd.openxmlformats-
      officedocument.wordprocessingml.document` If nothing specified, output
      files will be in the same format as the original file.
    GlossariesValue: Optional. Glossaries to be applied. It's keyed by target
      language code.
    ModelsValue: Optional. The models to use for translation. Map's key is
      target language code. Map's value is the model name. Value can be a
      built-in general model, or an AutoML Translation model. The value format
      depends on model type: - AutoML Translation models: `projects/{project-
      number-or-id}/locations/{location-id}/models/{model-id}` - General
      (built-in) models: `projects/{project-number-or-id}/locations/{location-
      id}/models/general/nmt`, If the map is empty or a specific model is not
      requested for a language pair, then default google model (nmt) is used.

  Fields:
    customizedAttribution: Optional. This flag is to support user customized
      attribution. If not provided, the default is `Machine Translated by
      Google`. Customized attribution should follow rules in
      https://cloud.google.com/translate/attribution#attribution_and_logos
    enableRotationCorrection: Optional. If true, enable auto rotation
      correction in DVS.
    enableShadowRemovalNativePdf: Optional. If true, use the text removal
      server to remove the shadow text on background image for native pdf
      translation. Shadow removal feature can only be enabled when
      is_translate_native_pdf_only: false && pdf_native_only: false
    formatConversions: Optional. File format conversion map to be applied to
      all input files. Map's key is the original mime_type. Map's value is the
      target mime_type of translated documents. Supported file format
      conversion includes: - `application/pdf` to
      `application/vnd.openxmlformats-
      officedocument.wordprocessingml.document` If nothing specified, output
      files will be in the same format as the original file.
    glossaries: Optional. Glossaries to be applied. It's keyed by target
      language code.
    inputConfigs: Required. Input configurations. The total number of files
      matched should be <= 100. The total content size to translate should be
      <= 100M Unicode codepoints. The files must use UTF-8 encoding.
    models: Optional. The models to use for translation. Map's key is target
      language code. Map's value is the model name. Value can be a built-in
      general model, or an AutoML Translation model. The value format depends
      on model type: - AutoML Translation models: `projects/{project-number-
      or-id}/locations/{location-id}/models/{model-id}` - General (built-in)
      models: `projects/{project-number-or-id}/locations/{location-
      id}/models/general/nmt`, If the map is empty or a specific model is not
      requested for a language pair, then default google model (nmt) is used.
    outputConfig: Required. Output configuration. If 2 input configs match to
      the same file (that is, same input path), we don't generate output for
      duplicate inputs.
    sourceLanguageCode: Required. The BCP-47 language code of the input
      document if known, for example, "en-US" or "sr-Latn". Supported language
      codes are listed in [Language
      Support](https://cloud.google.com/translate/docs/languages).
    targetLanguageCodes: Required. The BCP-47 language code to use for
      translation of the input document. Specify up to 10 language codes here.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FormatConversionsValue(_messages.Message):
        """Optional. File format conversion map to be applied to all input files.
    Map's key is the original mime_type. Map's value is the target mime_type
    of translated documents. Supported file format conversion includes: -
    `application/pdf` to `application/vnd.openxmlformats-
    officedocument.wordprocessingml.document` If nothing specified, output
    files will be in the same format as the original file.

    Messages:
      AdditionalProperty: An additional property for a FormatConversionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        FormatConversionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FormatConversionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GlossariesValue(_messages.Message):
        """Optional. Glossaries to be applied. It's keyed by target language
    code.

    Messages:
      AdditionalProperty: An additional property for a GlossariesValue object.

    Fields:
      additionalProperties: Additional properties of type GlossariesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a GlossariesValue object.

      Fields:
        key: Name of the additional property.
        value: A TranslateTextGlossaryConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TranslateTextGlossaryConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModelsValue(_messages.Message):
        """Optional. The models to use for translation. Map's key is target
    language code. Map's value is the model name. Value can be a built-in
    general model, or an AutoML Translation model. The value format depends on
    model type: - AutoML Translation models: `projects/{project-number-or-
    id}/locations/{location-id}/models/{model-id}` - General (built-in)
    models: `projects/{project-number-or-id}/locations/{location-
    id}/models/general/nmt`, If the map is empty or a specific model is not
    requested for a language pair, then default google model (nmt) is used.

    Messages:
      AdditionalProperty: An additional property for a ModelsValue object.

    Fields:
      additionalProperties: Additional properties of type ModelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ModelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customizedAttribution = _messages.StringField(1)
    enableRotationCorrection = _messages.BooleanField(2)
    enableShadowRemovalNativePdf = _messages.BooleanField(3)
    formatConversions = _messages.MessageField('FormatConversionsValue', 4)
    glossaries = _messages.MessageField('GlossariesValue', 5)
    inputConfigs = _messages.MessageField('BatchDocumentInputConfig', 6, repeated=True)
    models = _messages.MessageField('ModelsValue', 7)
    outputConfig = _messages.MessageField('BatchDocumentOutputConfig', 8)
    sourceLanguageCode = _messages.StringField(9)
    targetLanguageCodes = _messages.StringField(10, repeated=True)