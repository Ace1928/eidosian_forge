from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagQuery(_messages.Message):
    """A query to retrieve relevant contexts.

  Fields:
    similarityTopK: Optional. The number of contexts to retrieve.
    text: Optional. The query in text format to get relevant contexts.
  """
    similarityTopK = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    text = _messages.StringField(2)