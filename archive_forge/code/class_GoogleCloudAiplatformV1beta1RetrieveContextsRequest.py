from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RetrieveContextsRequest(_messages.Message):
    """Request message for VertexRagService.RetrieveContexts.

  Fields:
    query: Required. Single RAG retrieve query.
    vertexRagStore: The data source for Vertex RagStore.
  """
    query = _messages.MessageField('GoogleCloudAiplatformV1beta1RagQuery', 1)
    vertexRagStore = _messages.MessageField('GoogleCloudAiplatformV1beta1RetrieveContextsRequestVertexRagStore', 2)