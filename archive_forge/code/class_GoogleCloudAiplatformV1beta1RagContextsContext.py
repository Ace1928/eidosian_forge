from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagContextsContext(_messages.Message):
    """A context of the query.

  Fields:
    distance: The distance between the query vector and the context text
      vector.
    sourceUri: For vertex RagStore, if the file is imported from Cloud Storage
      or Google Drive, source_uri will be original file URI in Cloud Storage
      or Google Drive; if file is uploaded, source_uri will be file display
      name.
    text: The text chunk.
  """
    distance = _messages.FloatField(1)
    sourceUri = _messages.StringField(2)
    text = _messages.StringField(3)