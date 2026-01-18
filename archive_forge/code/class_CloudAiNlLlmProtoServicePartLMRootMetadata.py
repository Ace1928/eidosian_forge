from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePartLMRootMetadata(_messages.Message):
    """Metadata provides extra info for building the LM Root request.

  Fields:
    chunkId: Chunk id that will be used when mapping the part to the LM Root's
      chunk.
  """
    chunkId = _messages.StringField(1)