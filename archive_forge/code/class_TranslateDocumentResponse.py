from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateDocumentResponse(_messages.Message):
    """A translated document response message.

  Fields:
    documentTranslation: Translated document.
    glossaryConfig: The `glossary_config` used for this translation.
    glossaryDocumentTranslation: The document's translation output if a
      glossary is provided in the request. This can be the same as
      [TranslateDocumentResponse.document_translation] if no glossary terms
      apply.
    model: Only present when 'model' is present in the request. 'model' is
      normalized to have a project number. For example: If the 'model' field
      in TranslateDocumentRequest is: `projects/{project-
      id}/locations/{location-id}/models/general/nmt` then `model` here would
      be normalized to `projects/{project-number}/locations/{location-
      id}/models/general/nmt`.
  """
    documentTranslation = _messages.MessageField('DocumentTranslation', 1)
    glossaryConfig = _messages.MessageField('TranslateTextGlossaryConfig', 2)
    glossaryDocumentTranslation = _messages.MessageField('DocumentTranslation', 3)
    model = _messages.StringField(4)