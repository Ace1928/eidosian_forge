from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchTranslateTextRequest(_messages.Message):
    """The batch translation request.

  Messages:
    GlossariesValue: Optional. Glossaries to be applied for translation. It's
      keyed by target language code.
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.
      See https://cloud.google.com/translate/docs/labels for more information.
    ModelsValue: Optional. The models to use for translation. Map's key is
      target language code. Map's value is model name. Value can be a built-in
      general model, or an AutoML Translation model. The value format depends
      on model type: - AutoML Translation models: `projects/{project-number-
      or-id}/locations/{location-id}/models/{model-id}` - General (built-in)
      models: `projects/{project-number-or-id}/locations/{location-
      id}/models/general/nmt`, If the map is empty or a specific model is not
      requested for a language pair, then default google model (nmt) is used.

  Fields:
    glossaries: Optional. Glossaries to be applied for translation. It's keyed
      by target language code.
    inputConfigs: Required. Input configurations. The total number of files
      matched should be <= 100. The total content size should be <= 100M
      Unicode codepoints. The files must use UTF-8 encoding.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter. See
      https://cloud.google.com/translate/docs/labels for more information.
    models: Optional. The models to use for translation. Map's key is target
      language code. Map's value is model name. Value can be a built-in
      general model, or an AutoML Translation model. The value format depends
      on model type: - AutoML Translation models: `projects/{project-number-
      or-id}/locations/{location-id}/models/{model-id}` - General (built-in)
      models: `projects/{project-number-or-id}/locations/{location-
      id}/models/general/nmt`, If the map is empty or a specific model is not
      requested for a language pair, then default google model (nmt) is used.
    outputConfig: Required. Output configuration. If 2 input configs match to
      the same file (that is, same input path), we don't generate output for
      duplicate inputs.
    sourceLanguageCode: Required. Source language code.
    targetLanguageCodes: Required. Specify up to 10 language codes here.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GlossariesValue(_messages.Message):
        """Optional. Glossaries to be applied for translation. It's keyed by
    target language code.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModelsValue(_messages.Message):
        """Optional. The models to use for translation. Map's key is target
    language code. Map's value is model name. Value can be a built-in general
    model, or an AutoML Translation model. The value format depends on model
    type: - AutoML Translation models: `projects/{project-number-or-
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
    glossaries = _messages.MessageField('GlossariesValue', 1)
    inputConfigs = _messages.MessageField('InputConfig', 2, repeated=True)
    labels = _messages.MessageField('LabelsValue', 3)
    models = _messages.MessageField('ModelsValue', 4)
    outputConfig = _messages.MessageField('OutputConfig', 5)
    sourceLanguageCode = _messages.StringField(6)
    targetLanguageCodes = _messages.StringField(7, repeated=True)