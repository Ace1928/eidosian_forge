from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1Entity(_messages.Message):
    """Detected entity from video analysis.

  Fields:
    description: Textual description, e.g., `Fixed-gear bicycle`.
    entityId: Opaque entity ID. Some IDs may be available in [Google Knowledge
      Graph Search API](https://developers.google.com/knowledge-graph/).
    languageCode: Language code for `description` in BCP-47 format.
  """
    description = _messages.StringField(1)
    entityId = _messages.StringField(2)
    languageCode = _messages.StringField(3)