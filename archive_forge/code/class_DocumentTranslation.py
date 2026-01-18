from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentTranslation(_messages.Message):
    """A translated document message.

  Fields:
    byteStreamOutputs: The array of translated documents. It is expected to be
      size 1 for now. We may produce multiple translated documents in the
      future for other type of file formats.
    detectedLanguageCode: The detected language for the input document. If the
      user did not provide the source language for the input document, this
      field will have the language code automatically detected. If the source
      language was passed, auto-detection of the language does not occur and
      this field is empty.
    mimeType: The translated document's mime type.
  """
    byteStreamOutputs = _messages.BytesField(1, repeated=True)
    detectedLanguageCode = _messages.StringField(2)
    mimeType = _messages.StringField(3)