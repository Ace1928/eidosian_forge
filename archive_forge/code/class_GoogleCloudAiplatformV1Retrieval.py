from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Retrieval(_messages.Message):
    """Defines a retrieval tool that model can call to access external
  knowledge.

  Fields:
    disableAttribution: Optional. Disable using the result from this tool in
      detecting grounding attribution. This does not affect how the result is
      given to the model for generation.
    vertexAiSearch: Set to use data source powered by Vertex AI Search.
  """
    disableAttribution = _messages.BooleanField(1)
    vertexAiSearch = _messages.MessageField('GoogleCloudAiplatformV1VertexAISearch', 2)